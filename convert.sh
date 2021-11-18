#!/bin/bash

wget https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5m6.pt -P yolov5 -O yolov5m6.pt

cp tensorrtx/yolov5/gen_wts.py yolov5

cd /workspace/yolov5

# pip install -r requirements.txt

python gen_wts.py -w yolov5m6.pt -o yolov5m6.wts

cd /workspace/tensorrtx/yolov5

rm -rf build

mkdir -p build

cd build
# update CLASS_NUM in yololayer.h if your model is trained on custom dataset
cp /workspace/yolov5/yolov5m6.wts /workspace/tensorrtx/yolov5/build
cmake ..
make -j8

./yolov5 -s yolov5m6.wts yolov5m6.engine m6

exec "$@"