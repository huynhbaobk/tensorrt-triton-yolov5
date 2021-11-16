#include "layers.h"

IScaleLayer* addBatchNorm2d(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps)
{
    float* gamma = (float*)weightMap[lname + "gamma"].values; // scale
    float* beta = (float*)weightMap[lname + "beta"].values;   // offset
    float* mean = (float*)weightMap[lname + "moving_mean"].values;
    float* var = (float*)weightMap[lname + "moving_variance"].values;
    int len = weightMap[lname + "moving_variance"].count;

    float* scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (auto i = 0; i < len; i++)
    {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{ DataType::kFLOAT, scval, len };

    float* shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (auto i = 0; i < len; i++)
    {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{ DataType::kFLOAT, shval, len };

    float* pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (auto i = 0; i < len; i++)
    {
        pval[i] = 1.0;
    }
    Weights power{ DataType::kFLOAT, pval, len };

    IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    return scale_1;
}

IActivationLayer* bottleneck(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int ch, int stride, std::string lname, int branch_type)
{

    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

    IConvolutionLayer* conv1 = network->addConvolutionNd(input, ch, DimsHW{ 1, 1 }, weightMap[lname + "conv1/weights"], emptywts);
    assert(conv1);

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "conv1/BatchNorm/", 1e-5);
    assert(bn1);

    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    IConvolutionLayer* conv2 = network->addConvolutionNd(*relu1->getOutput(0), ch, DimsHW{ 3, 3 }, weightMap[lname + "conv2/weights"], emptywts);
    conv2->setStrideNd(DimsHW{ stride, stride });
    conv2->setPaddingNd(DimsHW{ 1, 1 });
    assert(conv2);

    IScaleLayer* bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + "conv2/BatchNorm/", 1e-5);
    assert(bn2);

    IActivationLayer* relu2 = network->addActivation(*bn2->getOutput(0), ActivationType::kRELU);
    assert(relu2);

    IConvolutionLayer* conv3 = network->addConvolutionNd(*relu2->getOutput(0), ch * 4, DimsHW{ 1, 1 }, weightMap[lname + "conv3/weights"], emptywts);
    assert(conv3);

    IScaleLayer* bn3 = addBatchNorm2d(network, weightMap, *conv3->getOutput(0), lname + "conv3/BatchNorm/", 1e-5);
    assert(bn3);
    IElementWiseLayer* ew1;
    // branch_type 0:shortcut,1:conv+bn+shortcut,2:maxpool+shortcut
    if (branch_type == 0)
    {
        ew1 = network->addElementWise(input, *bn3->getOutput(0), ElementWiseOperation::kSUM);
        assert(ew1);
    }
    else if (branch_type == 1)
    {
        IConvolutionLayer* conv4 = network->addConvolutionNd(input, ch * 4, DimsHW{ 1, 1 }, weightMap[lname + "shortcut/weights"], emptywts);
        conv4->setStrideNd(DimsHW{ stride, stride });
        assert(conv4);
        IScaleLayer* bn4 = addBatchNorm2d(network, weightMap, *conv4->getOutput(0), lname + "shortcut/BatchNorm/", 1e-5);
        assert(bn4);
        ew1 = network->addElementWise(*bn4->getOutput(0), *bn3->getOutput(0), ElementWiseOperation::kSUM);
        assert(ew1);
    }
    else
    {
        IPoolingLayer* pool = network->addPoolingNd(input, PoolingType::kMAX, DimsHW{ 1, 1 });
        pool->setStrideNd(DimsHW{ 2, 2 });
        assert(pool);
        ew1 = network->addElementWise(*pool->getOutput(0), *bn3->getOutput(0), ElementWiseOperation::kSUM);
        assert(ew1);
    }
    IActivationLayer* relu3 = network->addActivation(*ew1->getOutput(0), ActivationType::kRELU);
    assert(relu3);
    return relu3;
}

IActivationLayer* addConvRelu(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, int kernel, int stride, std::string lname)
{
    IConvolutionLayer* conv = network->addConvolutionNd(input, 256, DimsHW{ kernel, kernel }, weightMap[lname + "weights"], weightMap[lname + "biases"]);
    conv->setStrideNd(DimsHW{ stride, stride });
    if (kernel == 3)
    {
        conv->setPaddingNd(DimsHW{ 1, 1 });
    }
    assert(conv);

    IActivationLayer* ac = network->addActivation(*conv->getOutput(0), ActivationType::kRELU);
    assert(ac);
    return ac;
}