# OpenVINO-Darknet-YOLOv3
OpenVINO toolkit: tối đa hóa hiệu suất xử lí model Darknet-YOLOv3
OpenVINO toolkit: maximizing performance YOLOv3 tiny model workloads across Intel® hardware
## Resource & Reference:
Intel OpenVINO™: https://software.intel.com/en-us/openvino-toolkit
YOLOv3 Darket: https://pjreddie.com/darknet/
Tensorflow converter: https://github.com/mystic123/tensorflow-yolo-v3
My blog: https://ptitdeveloper.com/
## Configuration
### Full configuration guide: https://ptitdevdeloper.com/blog/openvino-toi-uu-hoa-hieu-suat-model-darknet-yolov3/
```bash
source  /opt/intel/openvino_2020.1.023/bin/setupvars.sh
cd /home/$USER/Desktop
git clone https://github.com/Namptiter/OpenVINO-Darknet-YOLOv3.git
cd OpenVINO-Darknet-YOLOv3
```
##### backup/ : YOLOv3 model (.weight)
##### video/ : input (.mp4)
##### tf_call_ie_layer/ : converter file (YOLOv3 -> tensorflow)
##### model_optimizer/ : converter file (tensorflow -> IR)
##### model/ : IR model (.bin, .xml)

### Convert YOLO model to Tensorflow model:
```bash
python3 tensorflow-yolo-v3/convert_weights_pb.py --class_names yolo.names --data_format NHWC --weights_file backup/yolov3-tiny_2.weights --tiny --size 832
```
### Convert Tensorflow model to IR model
```bash
python3 model_optimizer/mo_tf.py --input_model frozen_darknet_yolov3_model.pb --tensorflow_use_custom_operations_config yolo_v3_tiny.json --output_dir model -b 1
```
### Run Inference Engine demo with Python
```bash
python3 object_detection_demo_yolov3_async.py -m model/frozen_darknet_yolov3_model.xml -i video/v1.mp4 -d CPU -t 0.4
```
### Run Inference Engine with C++
```bash
cd /opt/intel/openvino_2020.1.023/deployment_tools/inference_engine/demos
./build_demos.sh
./home/$USER/omz_demos_build/intel64/Release/object_detection_demo_yolov3_async -i /home/$USER/Desktop/OpenVINO-Darknet-YOLOv3/video/v1.mp4 -m /home/$USER/Desktop/OpenVINO-Darknet-YOLOv3/model/frozen_darknet_yolov3_model.xml -d CPU -t 0.4
```
###### If I got any problem in copyright, Please contact to vuhoainam7121998@gmail.com or nam.vh@lophocvui.edu.vn
