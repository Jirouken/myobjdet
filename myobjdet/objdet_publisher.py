import rclpy
from rclpy.node import Node

from std_msgs.msg import String

import argparse
import cv2
import os
from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference

# import detect


class ObjDetPublisher(Node):

    def __init__(self):
        super().__init__('objdet_publisher')
        self.publisher_ = self.create_publisher(String, 'objdet', 10)
        timer_period = 0.5
        self.objdet = ObjDet()
        self.timer = self.create_timer(timer_period, self.timer_callback)
        

    def timer_callback(self):
        msg = String()
        msg.data = self.objdet.detect()

        self.publisher_.publish(msg)
        self.get_logger().info('%s' % msg.data)


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
                pass
            cv2_im = frame
            cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
            cv2_im_rgb = cv2.resize(cv2_im_rgb, self.inference_size)
            run_inference(self.interpreter, cv2_im_rgb.tobytes())
            objs = get_objects(self.interpreter, self.threshold)[:self.top_k]

            return get_label_score(objs, self.labels)

    def __del__(self):
        self.cap.release()


def main(args=None):
    rclpy.init(args=args)

    object_publisher = ObjDetPublisher()

    rclpy.spin(object_publisher)

    object_publisher.destroy_node()
    del object_publisher.objdet
    rclpy.shutdown()


if __name__ == '__main__':
    main()
