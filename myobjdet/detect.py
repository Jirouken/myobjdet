# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A demo that runs object detection on camera frames using OpenCV.

TEST_DATA=model

Run face detection model:
python3 detect.py \
  --model ${TEST_DATA}/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite

Run coco model:
python3 detect.py \
  --model ${TEST_DATA}/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite \
  --labels ${TEST_DATA}/coco_labels.txt

"""
import argparse
import cv2
import os

from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference


class ObjDet():
    def __init__(self):
        default_model_dir = 'model'
        default_model = 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
        default_labels = 'coco_labels.txt'    

        self.model = os.path.join(default_model_dir,default_model)
        self.labels = os.path.join(default_model_dir, default_labels)
        self.camera_idx = 0
        self.top_k = 3

        print('Loading {} with {} labels.'.format(self.model, self.labels))
        self.interpreter = make_interpreter(self.model)
        self.interpreter.allocate_tensors()
        self.labels = read_label_file(self.labels)
        self.inference_size = input_size(self.interpreter)
        self.threshold = 0.1

        self.cap = cv2.VideoCapture(self.camera_idx)

    def get_label_score(self, objs):
        if len(objs) >= 1:
            percent = int(100 * objs[0].score)
            label = '{}% {}'.format(percent, self.labels.get(objs[0].id, objs[0].id))
        else:
            percent = '--'
            label = '{}% {}'.format(percent, 'None')
        return label

    def detect(self):
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            cv2_im = frame
            cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
            cv2_im_rgb = cv2.resize(cv2_im_rgb, self.inference_size)
            run_inference(self.interpreter, cv2_im_rgb.tobytes())
            objs = get_objects(self.interpreter, self.threshold)[:self.top_k]

            return get_label_score(objs, self.labels)

        else:
            break

    def __del__(self):
        self.cap.release()
