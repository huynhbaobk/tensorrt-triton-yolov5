sudo docker run --name=yolov5_trt_convert --rm -it --gpus '"device=0"' -v ${PWD}:/workspace baohuynhbk/tensorrt-20.08-py3-opencv4:latest

# sudo docker run -it --gpus all -v ${PWD}:/workspace baohuynhbk/tensorrt-20.07-py3-opencv4:latest