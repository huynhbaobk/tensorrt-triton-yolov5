#!/usr/bin/env python

import argparse
import numpy as np
import sys
import cv2
import threading
import os
import shutil

import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

from processing import preprocess, postprocess
from render import render_box, render_filled_box, get_text_size, render_text, RAND_COLORS, plot_one_box
from labels import COCOLabels


def get_img_path_batches(batch_size, img_dir):
    ret = []
    batch = []
    for root, dirs, files in os.walk(img_dir):
        for name in files:
            if len(batch) == batch_size:
                ret.append(batch)
                batch = []
            batch.append(os.path.join(root, name))
    if len(batch) > 0:
        ret.append(batch)
    return ret

class inferThread(threading.Thread):
    def __init__(self, thread_num,
                         triton_client,
                         model_name,
                         inputs,
                         outputs,
                         client_timeout):
        threading.Thread.__init__(self)
        self.thread_num = thread_num
        self.triton_client = triton_client
        self.model_name = model_name
        self.inputs = inputs
        self.outputs = outputs
        self.client_timeout = client_timeout

    def run(self):
#        print(f"Run inference thread {self.thread_num}")
        for i in range(40):
            results = self.triton_client.infer(model_name=self.model_name,
                                inputs=self.inputs,
                                outputs=self.outputs,
                                client_timeout=self.client_timeout)
#        print("inference done")
        
#         for i, img_path in enumerate(self.image_path_batch):
#             parent, filename = os.path.split(img_path)
#             save_name = os.path.join('output', filename)
#             # Save image
#             cv2.imwrite(save_name, batch_image_raw[i])
#         print('input->{}, time->{:.2f}ms, saving into output/'.format(self.image_path_batch, use_time * 1000))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',
                        choices=['dummy', 'image', 'video'],
                        default='dummy',
                        help='Run mode. \'dummy\' will send an emtpy buffer to the server to test if inference works. \'image\' will process an image. \'video\' will process a video.')
    parser.add_argument('input',
                        type=str,
                        nargs='?',
                        help='Input file to load from in image or video mode')
    parser.add_argument('-m',
                        '--model',
                        type=str,
                        required=False,
                        default='yolov5',
                        help='Inference model name, default yolov5')
    parser.add_argument('--width',
                        type=int,
                        required=False,
                        default=640,
                        help='Inference model input width, default 608')
    parser.add_argument('--height',
                        type=int,
                        required=False,
                        default=640,
                        help='Inference model input height, default 608')
    parser.add_argument('-u',
                        '--url',
                        type=str,
                        required=False,
                        default='localhost:8221',
                        help='Inference server URL, default localhost:8001')
    parser.add_argument('-o',
                        '--out',
                        type=str,
                        required=False,
                        default='',
                        help='Write output into file instead of displaying it')
    parser.add_argument('-c',
                        '--confidence',
                        type=float,
                        required=False,
                        default=0.5,
                        help='Confidence threshold for detected objects, default 0.5')
    parser.add_argument('-n',
                        '--nms',
                        type=float,
                        required=False,
                        default=0.45,
                        help='Non-maximum suppression threshold for filtering raw boxes, default 0.45')
    parser.add_argument('-f',
                        '--fps',
                        type=float,
                        required=False,
                        default=24.0,
                        help='Video output fps, default 24.0 FPS')
    parser.add_argument('-i',
                        '--model-info',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Print model status, configuration and statistics')
    parser.add_argument('-v',
                        '--verbose',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose client output')
    parser.add_argument('-t',
                        '--client-timeout',
                        type=float,
                        required=False,
                        default=None,
                        help='Client timeout in seconds, default no timeout')
    parser.add_argument('-s',
                        '--ssl',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable SSL encrypted channel to the server')
    parser.add_argument('-r',
                        '--root-certificates',
                        type=str,
                        required=False,
                        default=None,
                        help='File holding PEM-encoded root certificates, default none')
    parser.add_argument('-p',
                        '--private-key',
                        type=str,
                        required=False,
                        default=None,
                        help='File holding PEM-encoded private key, default is none')
    parser.add_argument('-x',
                        '--certificate-chain',
                        type=str,
                        required=False,
                        default=None,
                        help='File holding PEM-encoded certicate chain default is none')

    FLAGS = parser.parse_args()
    print(FLAGS)
    # Create server context
    try:
        triton_client = grpcclient.InferenceServerClient(
            url=FLAGS.url,
            verbose=FLAGS.verbose,
            ssl=FLAGS.ssl,
            root_certificates=FLAGS.root_certificates,
            private_key=FLAGS.private_key,
            certificate_chain=FLAGS.certificate_chain)
    except Exception as e:
        print("context creation failed: " + str(e))
        sys.exit()

    # Health check
    if not triton_client.is_server_live():
        print("FAILED : is_server_live")
        sys.exit(1)

    if not triton_client.is_server_ready():
        print("FAILED : is_server_ready")
        sys.exit(1)
    
    if not triton_client.is_model_ready(FLAGS.model):
        print("FAILED : is_model_ready")
        sys.exit(1)

    if FLAGS.model_info:
        # Model metadata
        try:
            metadata = triton_client.get_model_metadata(FLAGS.model)
            print(metadata)
        except InferenceServerException as ex:
            if "Request for unknown model" not in ex.message():
                print("FAILED : get_model_metadata")
                print("Got: {}".format(ex.message()))
                sys.exit(1)
            else:
                print("FAILED : get_model_metadata")
                sys.exit(1)

        # Model configuration
        try:
            config = triton_client.get_model_config(FLAGS.model)
            if not (config.config.name == FLAGS.model):
                print("FAILED: get_model_config")
                sys.exit(1)
            print(config)
        except InferenceServerException as ex:
            print("FAILED : get_model_config")
            print("Got: {}".format(ex.message()))
            sys.exit(1)


    # IMAGE MODE
    if FLAGS.mode == 'image':
        print("Running in 'image' mode")
        if not FLAGS.input:
            print("FAILED: no input image")
            sys.exit(1)
        
        inputs = []
        outputs = []
        inputs.append(grpcclient.InferInput('data', [1, 3, FLAGS.width, FLAGS.height], "FP32"))
        outputs.append(grpcclient.InferRequestedOutput('prob'))

        print("Creating buffer from image file...")
        input_image = cv2.imread(str(FLAGS.input))
        if input_image is None:
            print(f"FAILED: could not load input image {str(FLAGS.input)}")
            sys.exit(1)
        input_image_buffer = preprocess(input_image, [FLAGS.width, FLAGS.height])
        input_image_buffer = np.expand_dims(input_image_buffer, axis=0)
        inputs[0].set_data_from_numpy(input_image_buffer)

        print("Invoking inference...")

#         if os.path.exists('output/'):
#             shutil.rmtree('output/')
#         os.makedirs('output/')
        
        NUM_THREADS = 30
        try:
            image_dir = "data/"
            threads_list = []
            
            for thread_num in range(NUM_THREADS):
                # create a new thread to do inference
                thread1 = inferThread(thread_num, 
                                      triton_client,
                                      model_name=FLAGS.model,
                                      inputs=inputs,
                                      outputs=outputs,
                                      client_timeout=FLAGS.client_timeout)
                threads_list.append(thread1)
            import time
            start_time = time.time()
            for thread1 in threads_list:
                thread1.start()
            
            for thread1 in threads_list:
                thread1.join()
            print("FPS: ",NUM_THREADS*40/(time.time()-start_time),(time.time()-start_time),NUM_THREADS*20)
        except:
            print("FAILED to infer model")
