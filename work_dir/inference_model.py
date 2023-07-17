from mmdeploy.apis.utils import build_task_processor
from mmdeploy.utils import get_input_shape, load_config
from PIL import Image
import torch
import numpy as np

model_cfg='/home/gideon/mmworks/mmdetection/configs/mask_rcnn/mask-rcnn_r101_fpn_1x_taco.py'
deploy_cfg='/home/gideon/mmworks/mmdeploy/configs/mmdet/instance-seg/instance-seg_tensorrt-fp16_dynamic-320x320-1344x1344.py'
backend_model = ['./end2end.engine']
img='/home/gideon/mmworks/mmdetection/demo/taco4.jpg'
device = "cuda:0"

# read deploy_cfg and model_cfg
deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)

# build task and backend model
task_processor = build_task_processor(model_cfg, deploy_cfg, device)
model = task_processor.build_backend_model(backend_model)

# process input image
input_shape = get_input_shape(deploy_cfg)
model_inputs, _ = task_processor.create_input(img, input_shape)

# do model inference
with torch.no_grad():
    result = model.test_step(model_inputs)

# Open img as numpy array
img_np = np.asarray(Image.open(img))

# visualize results
task_processor.visualize(
    image=img_np,
    model=model,
    result=result[0],
    window_name='visualize',
    output_file='output_detection3.png')

# TODO: test on all images (taco sea) 
