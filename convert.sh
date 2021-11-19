#!/bin/bash
cd /workspace

wget https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5m6.pt -P /workspace/yolov5 -O /workspace/yolov5/yolov5m6.pt

cp tensorrtx/yolov5/gen_wts.py yolov5

cd /workspace/yolov5

pip install -r requirements.txt

python /workspace/yolov5/gen_wts.py -w /workspace/yolov5/yolov5m6.pt -o /workspace/yolov5/yolov5m6.wts

cd /workspace/tensorrtx/yolov5

rm -rf /workspace/tensorrtx/yolov5/build

mkdir -p /workspace/tensorrtx/yolov5/build

cd /workspace/tensorrtx/yolov5/build
# update CLASS_NUM in yololayer.h if your model is trained on custom dataset
cp /workspace/yolov5/yolov5m6.wts /workspace/tensorrtx/yolov5/build
cmake ..
make -j8

/workspace/tensorrtx/yolov5/build/yolov5 -s /workspace/tensorrtx/yolov5/build/yolov5m6.wts /workspace/tensorrtx/yolov5/build/yolov5m6.engine m6

exec "$@"
