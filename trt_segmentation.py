"""Compute segmentation maps for images in the input folder.
"""
import os
import glob
import cv2
import argparse

import util.io

import sys
import torch

from torchvision.transforms import Compose
from dpt.models import DPTSegmentationModel
from dpt.transforms import Resize, NormalizeImage, PrepareForNet

from loguru import logger

import tensorrt as trt
import numpy as np 
import common
import time
import onnxruntime

DTYE = trt.float32
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(None, "")

class TensorRTYoloModel:
    def __init__(self, model_path):
        self.TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        self.trt_runtime = trt.Runtime(self.TRT_LOGGER)
        with open(model_path, 'rb') as f:
            self.engine = self.trt_runtime.deserialize_cuda_engine(f.read())
        #automatic buffer allocation process 
        self.inputs_buffer, self.outputs_buffer, self.bindings, self.stream = common.allocate_buffers(self.engine)
        self.context = self.engine.create_execution_context()
    def infer(self, frame):
        img_input_1d = frame.ravel()
        np.copyto(self.inputs_buffer[0].host, img_input_1d)
        prediction = common.do_inference_v2(self.context, 
                    bindings=self.bindings, inputs=self.inputs_buffer, outputs=self.outputs_buffer, stream=self.stream)
        return prediction



def run(input_path, output_path, model_path, model_type="dpt_hybrid", optimize=True):
    """Run segmentation network

    Args:
        input_path (str): path to input folder
        output_path (str): path to output folder
        model_path (str): path to saved model
    """
    
    net_w = net_h = 480

    transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            PrepareForNet(),
        ]
    )

    # get input
    img_names = glob.glob(os.path.join(input_path, "*"))
    num_images = len(img_names)

    # create output folder
    os.makedirs(output_path, exist_ok=True)

    #model_path = "../dpt_seg_hyb_171121.trt"
    #dpt_engine = TensorRTYoloModel(model_path) 
    #model_path = "/media/31A079936F39FBF9/romil/onnx_cache_trt/dpt_seg_hyb_view_folded_2_161121.onnx"
    model_path = "/media/31A079936F39FBF9/romil/onnx_cache_trt/dpt_seg_hyb_view_folded_pad_resize_161121.onnx"
    session = onnxruntime.InferenceSession(model_path)   

    print("start processing")

    for ind, img_name in enumerate(img_names):

        print("  processing {} ({}/{})".format(img_name, ind + 1, num_images))

        # input
        img = util.io.read_image(img_name)
        img_input = transform({"image": img})["image"]

        # compute
        ort_inputs = {session.get_inputs()[0].name: img_input[None, :, :, :]}
        t2 = time.time()
        out = session.run(None, ort_inputs)
        t3 = time.time()
        prediction = out[0].copy()
        #prediction.resize((150, img.shape[0], img.shape[1]))
        prediction = torch.from_numpy(prediction)
        prediction = torch.nn.functional.interpolate(
                prediction, size=img.shape[:2], mode="bicubic", align_corners=False
            )
        print("prediction shape", prediction.shape, prediction.dtype)
        #prediction = np.argmax(prediction, axis=0) + 1
        prediction = torch.argmax(prediction, dim=1) + 1
        print("prediction shape after argmax", prediction.shape, prediction.dtype)
        prediction = prediction.squeeze().cpu().numpy()
        # output
        filename = os.path.join(
            output_path, os.path.splitext(os.path.basename(img_name))[0]
        )
        #print(prediction)
        #cv2.imwrite("prediction.png", prediction)
        util.io.write_segm_img(filename, img, prediction, alpha=0.5)

    print("finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i", "--input_path", default="input", help="folder with input images"
    )

    parser.add_argument(
        "-o", "--output_path", default="output_semseg", help="folder for output images"
    )

    parser.add_argument(
        "-m",
        "--model_weights",
        default=None,
        help="path to the trained weights of model",
    )

    # 'vit_large', 'vit_hybrid'
    parser.add_argument("-t", "--model_type", default="dpt_hybrid", help="model type")

    parser.add_argument("--optimize", dest="optimize", action="store_true")
    parser.add_argument("--no-optimize", dest="optimize", action="store_false")
    parser.set_defaults(optimize=True)

    args = parser.parse_args()

    default_models = {
        "dpt_large": "weights/dpt_large-ade20k-b12dca68.pt",
        "dpt_hybrid": "weights/dpt_hybrid-ade20k-53898607.pt",
    }

    if args.model_weights is None:
        args.model_weights = default_models[args.model_type]

    # compute segmentation maps
    run(
        args.input_path,
        args.output_path,
        args.model_weights,
        args.model_type,
        args.optimize,
    )
