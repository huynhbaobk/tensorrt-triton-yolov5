import numpy as np
import cv2
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

from processing import preprocess, postprocess
from render import render_box, render_filled_box, get_text_size, render_text, RAND_COLORS, plot_one_box
from labels import COCOLabels, STATUS

class Trion_grpc_infer_OD:
    def __init__(self, mode='image',
                       model='yolov5',
                       width=640, height=640,
                       url='localhost:8221',
                       confidence=0.5,
                       nms=0.45,
                       model_info=False,
                       verbose=False,
                       client_timeout=None,
                       ssl=False,
                       root_certificates=None,
                       private_key=None,
                       certificate_chain=None
                        ):
        
        ### Init triton client params
        self.mode = mode
        self.model = model
        self.width = width
        self.height = height
        self.url = url
        self.confidence = confidence
        self.nms = nms
        self.model_info = model_info
        self.verbose = verbose
        self.client_timeout = client_timeout
        self.ssl = ssl
        self.root_certificates = root_certificates
        self.private_key = private_key
        self.certificate_chain = certificate_chain

        self.batch_size = 1
        self.channels = 3
        self.data_type =  "FP32"

        ### Create triton client
        self.triton_client = self.create_grpc_client()

    def create_grpc_client(self):
        ### Create gRPC client for communicating with the server
        try:
            triton_client = grpcclient.InferenceServerClient(
                url=self.url,
                verbose=self.verbose,
                ssl=self.ssl,
                root_certificates=self.root_certificates,
                private_key=self.private_key,
                certificate_chain=self.certificate_chain)
        except Exception as e:
            print("Triton server context creation failed: " + str(e))
            return None


        ### Health check
        if not triton_client.is_server_live():
            print("FAILED : is_server_live")
            return None

        if not triton_client.is_server_ready():
            print("FAILED : is_server_ready")
            return None
        
        if not triton_client.is_model_ready(self.model):
            print("FAILED : is_model_ready")
            return None
    
        ### Print info
        if self.model_info:
            # Model metadata
            try:
                metadata = triton_client.get_model_metadata(self.model)
                print(metadata)
            except InferenceServerException as ex:
                if "Request for unknown model" not in ex.message():
                    print("FAILED : get_model_metadata")
                    print("Got: {}".format(ex.message()))
                else:
                    print("FAILED : get_model_metadata")

            # Model configuration
            try:
                config = triton_client.get_model_config(self.model)
                if not (config.config.name == self.model):
                    print("FAILED: get_model_config")
                print(config)
            except InferenceServerException as ex:
                print("FAILED : get_model_config")
                print("Got: {}".format(ex.message()))
        
        return triton_client

    def do_inference_sync(self, input_image):
        print("Running in 'image' mode")
        inputs = []
        outputs = []
        if input_image is None:
            print("FAILED: no input image")
            return STATUS.INVALID_ARGUMENT, None, None, None, None 

        inputs.append(grpcclient.InferInput('data', [self.batch_size, self.channels, self.width, self.height], self.data_type))
        outputs.append(grpcclient.InferRequestedOutput('prob'))

        print("Creating buffer from image file...")

        input_image_buffer = preprocess(input_image, [self.width, self.height])
        input_image_buffer = np.expand_dims(input_image_buffer, axis=0)
        inputs[0].set_data_from_numpy(input_image_buffer)

        print("Invoking inference...")
        results = self.triton_client.infer(model_name=self.model,
                                    inputs=inputs,
                                    outputs=outputs,
                                    client_timeout=self.client_timeout)
        if self.model_info:
            statistics = self.triton_client.get_inference_statistics(model_name=self.model)
            if len(statistics.model_stats) != 1:
                print("FAILED: get_inference_statistics")
            print(statistics)
        print("Done")

        result = results.as_numpy('prob')
        np.save('result_pred.npy', result)

        print(f"Received result buffer of size {result.shape}")
        print(f"Naive buffer sum: {np.sum(result)}")
        print("Inference Done!")

        status, num_box, classes, score, boxes = postprocess(result, input_image.shape[1], input_image.shape[0], [self.width, self.height], self.confidence, self.nms)

        # ### Print out result (For test)
        # for i in range(num_box):
        #     print(f"{COCOLabels(classes[i]).name}: {score[i]}")
        #     plot_one_box(boxes[i], input_image, color=tuple(RAND_COLORS[classes[i] % 64].tolist()), \
        #                                         label=f"{COCOLabels(classes[i]).name}:{score[i]:.2f}",)
        # cv2.imshow('image', input_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return status, num_box, classes, score, boxes

    def plot_bboxes(self, input_image, num_box, classes, score, boxes, save_path=None):
        for i in range(num_box):
            print(f"{COCOLabels(classes[i]).name}: {score[i]}")
            plot_one_box(boxes[i], input_image, color=tuple(RAND_COLORS[classes[i] % 64].tolist()), \
                                                label=f"{COCOLabels(classes[i]).name}:{score[i]:.2f}",)

        if save_path != None:
            cv2.imwrite(save_path, input_image)
            print(f"Saved result to {save_path}")

        return input_image


if __name__ == '__main__':
    client = Trion_grpc_infer_OD(url='10.10.37.119:8221')
    input_image = cv2.imread("./data/dog.jpg")
    status, num_box, classes, score, boxes = client.do_inference_sync(input_image)
    client.plot_bboxes(input_image, num_box, classes, score, boxes, "./data/dog_new.jpg")
