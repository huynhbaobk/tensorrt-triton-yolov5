import argparse
import numpy as np
import sys
import cv2
import pickle
import json

import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

from processing import preprocess, postprocess, postprocess_test
from render import render_box, render_filled_box, get_text_size, render_text, RAND_COLORS, plot_one_box
from labels import COCOLabels

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

    # DUMMY MODE
    if FLAGS.mode == 'dummy':
        print("Running in 'dummy' mode")
        print("Creating emtpy buffer filled with ones...")
        inputs = []
        outputs = []
        inputs.append(grpcclient.InferInput('data', [1, 3, FLAGS.width, FLAGS.height], "FP32"))
        inputs[0].set_data_from_numpy(np.ones(shape=(1, 3, FLAGS.width, FLAGS.height), dtype=np.float32))
        outputs.append(grpcclient.InferRequestedOutput('prob'))

        print("Invoking inference...")
        results = triton_client.infer(model_name=FLAGS.model,
                                    inputs=inputs,
                                    outputs=outputs,
                                    client_timeout=FLAGS.client_timeout)
        if FLAGS.model_info:
            statistics = triton_client.get_inference_statistics(model_name=FLAGS.model)
            if len(statistics.model_stats) != 1:
                print("FAILED: get_inference_statistics")
                sys.exit(1)
            print(statistics)
        print("Done")

        result = results.as_numpy('prob')
        print(f"Received result buffer of size {result.shape}")
        print(f"Naive buffer sum: {np.sum(result)}")

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
        results = triton_client.infer(model_name=FLAGS.model,
                                    inputs=inputs,
                                    outputs=outputs,
                                    client_timeout=FLAGS.client_timeout)

        if FLAGS.model_info:
            statistics = triton_client.get_inference_statistics(model_name=FLAGS.model)
            if len(statistics.model_stats) != 1:
                print("FAILED: get_inference_statistics")
                sys.exit(1)
            print(statistics)
        print("Done")

        result = results.as_numpy('prob')
        print(f"Received result buffer of size {result.shape}")
        print(f"Naive buffer sum: {np.sum(result)}")
        

#         with open('out', 'w') as f:
#             json.dump(results, f, allow_nan = True)

        detected_objects = postprocess_test(result, input_image.shape[1], input_image.shape[0], [FLAGS.width, FLAGS.height], FLAGS.confidence, FLAGS.nms)
        print(f"Detected objects: {len(detected_objects)}")

        for box in detected_objects:
            print(f"{COCOLabels(box.classID).name}: {box.confidence}")
            plot_one_box(box.box(), input_image,color=tuple(RAND_COLORS[box.classID % 64].tolist()), label=f"{COCOLabels(box.classID).name}:{box.confidence:.2f}",)   

        if FLAGS.out:
            cv2.imwrite(FLAGS.out, input_image)
            print(f"Saved result to {FLAGS.out}")
        else:
            cv2.imshow('image', input_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    # VIDEO MODE
    if FLAGS.mode == 'video':
        print("Running in 'video' mode")
        if not FLAGS.input:
            print("FAILED: no input video")
            sys.exit(1)

        inputs = []
        outputs = []
        inputs.append(grpcclient.InferInput('data', [1, 3, FLAGS.width, FLAGS.height], "FP32"))
        outputs.append(grpcclient.InferRequestedOutput('prob'))

        print("Opening input video stream...")
        cap = cv2.VideoCapture(FLAGS.input)
        if not cap.isOpened():
            print(f"FAILED: cannot open video {FLAGS.input}")
            sys.exit(1)

        counter = 0
        out = None
        print("Invoking inference...")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("failed to fetch next frame")
                break

            if counter == 0 and FLAGS.out:
                print("Opening output video stream...")
                fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
                out = cv2.VideoWriter(FLAGS.out, fourcc, FLAGS.fps, (frame.shape[1], frame.shape[0]))

            input_image_buffer = preprocess(frame, [FLAGS.width, FLAGS.height])
            input_image_buffer = np.expand_dims(input_image_buffer, axis=0)
            inputs[0].set_data_from_numpy(input_image_buffer)

            results = triton_client.infer(model_name=FLAGS.model,
                                    inputs=inputs,
                                    outputs=outputs,
                                    client_timeout=FLAGS.client_timeout)

            result = results.as_numpy('prob')

            detected_objects = postprocess_test(result, frame.shape[1], frame.shape[0], [FLAGS.width, FLAGS.height], FLAGS.confidence, FLAGS.nms)
            print(f"Frame {counter}: {len(detected_objects)} objects")
            counter += 1

            for box in detected_objects:
                print(f"{COCOLabels(box.classID).name}:{box.confidence}")
                plot_one_box(box.box(), input_image,color=tuple(RAND_COLORS[box.classID % 64].tolist()), label=f"{COCOLabels(box.classID).name}: {box.confidence:.2f}",)   

            if FLAGS.out:
                out.write(frame)
            else:
                cv2.imshow('image', frame)
                if cv2.waitKey(1) == ord('q'):
                    break

        if FLAGS.model_info:
            statistics = triton_client.get_inference_statistics(model_name=FLAGS.model)
            if len(statistics.model_stats) != 1:
                print("FAILED: get_inference_statistics")
                sys.exit(1)
            print(statistics)
        print("Done")

        cap.release()
        if FLAGS.out:
            out.release()
        else:
            cv2.destroyAllWindows()
        print("Done")
