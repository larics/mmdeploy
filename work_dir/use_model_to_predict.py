from mmdeploy.apis import inference_model

result = inference_model(
  model_cfg='/home/gideon/mmworks/mmdetection/configs/mask_rcnn/mask-rcnn_r101_fpn_1x_taco.py',
  deploy_cfg='/home/gideon/mmworks/mmdeploy/configs/mmdet/instance-seg/instance-seg_tensorrt-fp16_dynamic-320x320-1344x1344.py',
  backend_files=['end2end.engine'],
  img='/home/gideon/mmworks/mmdetection/demo/taco1.jpg',
  device='cuda:0')



print(result)
