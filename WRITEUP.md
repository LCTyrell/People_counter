# Project Write-Up

![people_counter](./images/people_counter.gif)

## Explaining Custom Layers

The process behind converting custom layers involves...

Some of the potential reasons for handling custom layers are...

## Comparing Model Performance

A lot of models was tested in order to find one that have an accuracy sufficient for the project, and if possible, equivalent or superior to the Openvino pre-trained models.

Below the benchmark of the tests:

<table>
  <tr>
    <th rowspan="2">Framework</th>
    <th rowspan="2">Model</th>
    <th colspan="3">Inference time (ms)</th>
    <th rowspan="2">Accuracy</th>
    <th colspan="3">FPS</th>
  </tr>
  <tr>
    <th>CPU</th>
    <th>GPU</th>
    <th>VPU</th>    
    <th>CPU</th>
    <th>GPU</th>
    <th>VPU</th>
  </tr>
  <tr>
    <td>Pytorch</td>
    <td>Yolo v3</td>
    <td>84</td>
    <td>-</td>
    <td>-</td>    
    <td>High</td>
    <td>12</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>Openvino (TF)</td>
    <td>SSD Inception v2 coco</td>
    <td>22</td>
    <td>41</td>
    <td>75</td>
    <td>Low</td>
    <td>45</td>
    <td>24</td>
    <td>13</td>
  </tr>
  <tr>
  <td>Openvino (TF)</td>
  <td>SSD Mobilenet v1 coco</td>
  <td>254</td>
  <td>-</td>
  <td>-</td>
  <td>Low</td>
  <td>4</td>
  <td>-</td>
  <td>-</td>
  </tr>
  <tr>
  <td>Openvino</td>
  <td>Retail-0013</td>
  <td>8</td>
  <td>12</td>
  <td>129</td>
  <td>High</td>
  <td>125</td>
  <td>83</td>
  <td>8</td>
  </tr>
  <tr>
  <td>Openvino (TF)</td>
  <td>SSD Resnet 50</td>
  <td>356</td>
  <td>-</td>
  <td>-</td>
  <td>Very Low</td>
  <td>3</td>
  <td>-</td>
  <td>-</td>
  </tr>
  <tr>
  <td>Openvino (TF)</td>
  <td>SSD Mobilenet v2 coco</td>
  <td>12</td>
  <td>24</td>
  <td>63</td>
  <td>Low</td>
  <td>83</td>
  <td>41</td>
  <td>16</td>
  </tr>
  <tr>
  <td>Openvino (TF)</td>
  <td>Yolo v3</td>
  <td>40-130 (1)</td>
  <td>20-380 (1)</td>
  <td>24-192 (1)</td>
  <td>Low</td>
  <td><10</td>
  <td><10</td>
  <td><10</td>
  </tr>
  <tr>
  <td>Openvino (TF)</td>
  <td>TinyYolo v3</td>
  <td>21</td>
  <td>22</td>
  <td>61</td>
  <td>Very Low</td>
  <td>48</td>
  <td>45</td>
  <td>16</td>
  </tr>
</table>

**The benchmark was realised on an i7 8700 CPU, an HD630 iGPU and an Neural Compute Stick (NCS2) VPU.**  

Only Yolo v3 on pytorch meet the requirements. It was slower than the Openvino Retail-0013 but the FPS is still suficient (the video rate of the demo run at 10 FPS).
Neverseless, sadly Yolo v3 Pytorch model can't be converted (at least for the 2019.3 version of Openvino).

A Tensorflow version of Yolo was converted and tested, but the result was deceptive : the accuracy gone from High to Low, and the FPS was well under the 10 FPS.

As verification, the demo provided by intel for Yolo was tested but the result was the same.

(1) The variability of the results given for Yolo comes to the difference in mesurement place (taking into account pre/post processing or not). The slowing factor seems not to come from the inference time but from the pre and post processing (certainly du to the complexity of the model). There certainly improvement possible on this way.

So, in definitive, the Retail-0013 model was selected. This model was trained and optimized (from SSD mobilenet v2) for the task and show a very good accuracy/FPS rate.

**Model size modification by openvino optimizer process:**
<table>
  <tr>
    <th >Model</th>
    <th >Before</th>
    <th >After</th>

  </tr>
  <tr>
    <td>Yolo v3</td>
    <td>248.2 Mo</td>
    <td>247.7 Mo</td>
  </tr>
  <tr>
    <td>SSD Resnet 50</td>
    <td>134.3 Mo</td>
    <td>206.7 Mo</td>
  </tr>
  <tr>
    <td>SSD Mobilenet v2 coco</td>
    <td>69.7 Mo</td>
    <td>67.3 Mo</td>
  </tr>
</table>

**Advantage of running AI to the EDGE (vs Cloud):**
* Cost saving: for service running 24/7 (videosurveillance), the cost is higly reduced
* No need costly network (in term of budget and computer ressource consumption)
* Better security of data

## Assess Model Use Cases

Some of the potential use cases of the people counter app are :

* **Smart city** : Implanting temporary or definitive camera to ameliorate infrastructure and services. (e.g. Ameliorate pieton circulation). This kind of app permit to have more data processed automaticaly and in real time.

* **Marketing** : For retail it could help to adapt to the flux of customers in a more reactive way and also stock data to make predictive models.


## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects are:
* Big changement in lightning could affect the model accuracy. If possible it must be avoided. If it can't be avoided the model could be adapted by a specific training to avoid lost of accuracy.
* Model accuracy depend on the customers requirement : accuracy or budget priority? speed or better accuracy?... all question that must be taken in account at the beginning of the prject.
* Camera choice will affect model acuracy and computation requierement, as well as budget. As for model accuracy, all this effect must be taken in account shortly in the project.

## Model Research

In investigating potential people counter models, I tried each of the following three models:

- Model 1: **Yolo v3**
  - Model Source: https://pjreddie.com/media/files/yolov3.weights
  - I converted the model to an Intermediate Representation with the following arguments:
        `python3 mo_tf.py
        --input_model <path_to>/yolo_v3.pb
        --tensorflow_use_custom_operations_config <path_to>/yolo_v3.json
        --batch 1`

  - The model was insufficient for the app because of his low accuracy and low FPS rate.
  - I tried to improve the model for the app by modifying accuracy threshold and testing different platform (CPU, GPU, VPU).

- Model 2: **SSD Resnet 50**
  - Model Source: http://download.tensorflow.org/models/object_detection/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz
  - I converted the model to an Intermediate Representation with the following arguments:
        `python <path_to>/mo.py --input_model <path_to>/frozen_inference_graph.pb \
        --tensorflow_object_detection_api_pipeline_config <path_to>/pipeline.config \
        --reverse_input_channels \
        --tensorflow_use_custom_operations_config <path_to>/ssd_v2_support.json`

  - The model was insufficient for the app because of his very low accuracy and inacurate FPS rate.
  - I tried to improve the model for the app by modifying accuracy threshold and testing different platform (CPU, GPU, VPU).

- Model 3: **SSD Mobilenet v2**
  - Model Source: http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
  - I converted the model to an Intermediate Representation with the following arguments:
        `python <path_to>/mo.py --input_model <path_to>/frozen_inference_graph.pb \
        --tensorflow_object_detection_api_pipeline_config <path_to>/pipeline.config \
        --reverse_input_channels \
        --tensorflow_use_custom_operations_config <path_to>/ssd_v2_support.json`

  - The model was insufficient for the app because of his low accuracy.
  - I tried to improve the model for the app by modifying accuracy threshold and testing different platform (CPU, GPU, VPU).
