#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include <cmath>

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

// stuff we know about the network and the input/output blobs
static const int INPUT_H = 224;
static const int INPUT_W = 224;
static const int OUTPUT_SIZE = 1000;

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";

using namespace nvinfer1;

static Logger gLogger;

// Load weights from files shared with TensorRT samples.
// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
std::map<std::string, Weights> loadWeights(const std::string file)
{
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file.");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--)
    {
        Weights wt{DataType::kFLOAT, nullptr, 0};
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        // Load blob
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x)
        {
            input >> std::hex >> val[x];
        }
        wt.values = val;
        
        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;
}

IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps) {
    float *gamma = (float*)weightMap[lname + ".weight"].values;
    float *beta = (float*)weightMap[lname + ".bias"].values;
    float *mean = (float*)weightMap[lname + ".running_mean"].values;
    float *var = (float*)weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".running_var"].count;
    std::cout << "len " << len << std::endl;

    float *scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{DataType::kFLOAT, scval, len};
    
    float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{DataType::kFLOAT, shval, len};

    float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        pval[i] = 1.0;
    }
    Weights power{DataType::kFLOAT, pval, len};

    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;
    IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    return scale_1;
}

ILayer* invertedRes(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname,
        int inch, int outch, int ksize, int s, int exp) {
    std::cout << lname << std::endl;
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    int midch = inch * exp;
    IConvolutionLayer* conv1 = network->addConvolutionNd(input, midch, DimsHW{1, 1}, weightMap[lname + "layers.0.weight"], emptywts);
    assert(conv1);
    IScaleLayer *bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "layers.1", 1e-5);
    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);
    IConvolutionLayer* conv2 = network->addConvolutionNd(*relu1->getOutput(0), midch, DimsHW{ksize, ksize}, weightMap[lname + "layers.3.weight"], emptywts);
    assert(conv2);
    conv2->setStrideNd(DimsHW{s, s});
    conv2->setPaddingNd(DimsHW{ksize / 2, ksize / 2});
    conv2->setNbGroups(midch);
    IScaleLayer *bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + "layers.4", 1e-5);
    IActivationLayer* relu2 = network->addActivation(*bn2->getOutput(0), ActivationType::kRELU);
    assert(relu2);
    IConvolutionLayer* conv3 = network->addConvolutionNd(*relu2->getOutput(0), outch, DimsHW{1, 1}, weightMap[lname + "layers.6.weight"], emptywts);
    assert(conv3);
    IScaleLayer *bn3 = addBatchNorm2d(network, weightMap, *conv3->getOutput(0), lname + "layers.7", 1e-5);

    if (inch == outch && s == 1) {
        IElementWiseLayer* ew1 = network->addElementWise(*bn3->getOutput(0), input, ElementWiseOperation::kSUM);
        assert(ew1);
        return ew1;
    }
    return bn3;
}

// Creat the engine using only the API and not any parser.
ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt)
{
    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Create input tensor of shape { 1, 1, 32, 32 } with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{3, INPUT_H, INPUT_W});
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights("../mnasnet.wts");
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    IConvolutionLayer* conv1 = network->addConvolutionNd(*data, 32, DimsHW{3, 3}, weightMap["layers.0.weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{2, 2});
    conv1->setPaddingNd(DimsHW{1, 1});
    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), "layers.1", 1e-5);
    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);
    IConvolutionLayer* conv2 = network->addConvolutionNd(*relu1->getOutput(0), 32, DimsHW{3, 3}, weightMap["layers.3.weight"], emptywts);
    assert(conv2);
    conv2->setPaddingNd(DimsHW{1, 1});
    conv2->setNbGroups(32);
    IScaleLayer* bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), "layers.4", 1e-5);
    IActivationLayer* relu2 = network->addActivation(*bn2->getOutput(0), ActivationType::kRELU);
    assert(relu2);
    IConvolutionLayer* conv3 = network->addConvolutionNd(*relu2->getOutput(0), 16, DimsHW{1, 1}, weightMap["layers.6.weight"], emptywts);
    assert(conv3);
    IScaleLayer* bn3 = addBatchNorm2d(network, weightMap, *conv3->getOutput(0), "layers.7", 1e-5);

    ILayer* ir1 = invertedRes(network, weightMap, *bn3->getOutput(0), "layers.8.0.", 16, 16, 3, 2, 3);
    ir1 = invertedRes(network, weightMap, *ir1->getOutput(0), "layers.8.1.", 16, 16, 3, 1, 3);
    ir1 = invertedRes(network, weightMap, *ir1->getOutput(0), "layers.8.2.", 16, 16, 3, 1, 3);
    ir1 = invertedRes(network, weightMap, *ir1->getOutput(0), "layers.9.0.", 16, 24, 5, 2, 3);
    ir1 = invertedRes(network, weightMap, *ir1->getOutput(0), "layers.9.1.", 24, 24, 5, 1, 3);
    ir1 = invertedRes(network, weightMap, *ir1->getOutput(0), "layers.9.2.", 24, 24, 5, 1, 3);
    ir1 = invertedRes(network, weightMap, *ir1->getOutput(0), "layers.10.0.", 24, 40, 5, 2, 6);
    ir1 = invertedRes(network, weightMap, *ir1->getOutput(0), "layers.10.1.", 40, 40, 5, 1, 6);
    ir1 = invertedRes(network, weightMap, *ir1->getOutput(0), "layers.10.2.", 40, 40, 5, 1, 6);
    ir1 = invertedRes(network, weightMap, *ir1->getOutput(0), "layers.11.0.", 40, 48, 3, 1, 6);
    ir1 = invertedRes(network, weightMap, *ir1->getOutput(0), "layers.11.1.", 48, 48, 3, 1, 6);
    ir1 = invertedRes(network, weightMap, *ir1->getOutput(0), "layers.12.0.", 48, 96, 5, 2, 6);
    ir1 = invertedRes(network, weightMap, *ir1->getOutput(0), "layers.12.1.", 96, 96, 5, 1, 6);
    ir1 = invertedRes(network, weightMap, *ir1->getOutput(0), "layers.12.2.", 96, 96, 5, 1, 6);
    ir1 = invertedRes(network, weightMap, *ir1->getOutput(0), "layers.12.3.", 96, 96, 5, 1, 6);
    ir1 = invertedRes(network, weightMap, *ir1->getOutput(0), "layers.13.0.", 96, 160, 3, 1, 6);

    IConvolutionLayer* conv4 = network->addConvolutionNd(*ir1->getOutput(0), 1280, DimsHW{1, 1}, weightMap["layers.14.weight"], emptywts);
    assert(conv4);
    IScaleLayer* bn4 = addBatchNorm2d(network, weightMap, *conv4->getOutput(0), "layers.15", 1e-5);
    IActivationLayer* relu3 = network->addActivation(*bn4->getOutput(0), ActivationType::kRELU);
    assert(relu3);
    IPoolingLayer* pool2 = network->addPoolingNd(*relu3->getOutput(0), PoolingType::kAVERAGE, DimsHW{7, 7});
    assert(pool2);

    IFullyConnectedLayer* fc1 = network->addFullyConnected(*pool2->getOutput(0), 1000, weightMap["classifier.1.weight"], weightMap["classifier.1.bias"]);
    assert(fc1);

    fc1->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    std::cout << "set name out" << std::endl;
    network->markOutput(*fc1->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(1 << 20);
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "build out" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*) (mem.second.values));
    }

    return engine;
}

void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream)
{
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = createEngine(maxBatchSize, builder, config, DataType::kFLOAT);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
    config->destroy();
}

void doInference(IExecutionContext& context, float* input, float* output, int batchSize)
{
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./mnasnet -s   // serialize model to plan file" << std::endl;
        std::cerr << "./mnasnet -d   // deserialize plan file and run inference" << std::endl;
        return -1;
    }

    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};

    if (std::string(argv[1]) == "-s") {
        IHostMemory* modelStream{nullptr};
        APIToModel(1, &modelStream);
        assert(modelStream != nullptr);

        std::ofstream p("mnasnet.engine", std::ios::binary);
        if (!p)
        {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        return 1;
    } else if (std::string(argv[1]) == "-d") {
        std::ifstream file("mnasnet.engine", std::ios::binary);
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }
    } else {
        return -1;
    }


    // Subtract mean from image
    float data[3 * INPUT_H * INPUT_W];
    for (int i = 0; i < 3 * INPUT_H * INPUT_W; i++)
        data[i] = 1.0;

    //PluginFactory pf;
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;

    // Run inference
    float prob[OUTPUT_SIZE];
    for (int i = 0; i < 100; i++) {
        auto start = std::chrono::system_clock::now();
        doInference(*context, data, prob, 1);
        auto end = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    }

    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    // Print histogram of the output distribution
    std::cout << "\nOutput:\n\n";
    for (unsigned int i = 0; i < OUTPUT_SIZE; i++)
    {
        std::cout << prob[i] << ", ";
        if (i % 10 == 0) std::cout << i / 10 << std::endl;
    }
    std::cout << std::endl;

    return 0;
}
