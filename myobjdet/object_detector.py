import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
import cv2
import os
from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference


class ObjectDetector(Node):

    def __init__(self):
        super().__init__('object_detector')
        self.objdet = ObjDet()
        self.create_subscription(
            Image,
            '/image_raw',
            self.cb_sub,
            10)
        self.publisher_ = self.create_publisher(String, "object_label", 10)
        timer_period = 0.5
        self.timer = self.create_timer(timer_period, self.cb_pub)
        self.msg_ = None

    def cb_sub(self, msg):
        self.msg_ = msg

    def cb_pub(self):
        try:
            bridge = CvBridge()
            img = bridge.imgmsg_to_cv2(self.msg_, "bgr8")
            res = self.objdet.detect(img)        
            msg = String()
            msg.data = res
            self.publisher_.publish(msg)
        except Exception as err:
            self.get_logger().info("err")

class ObjDet():
    def __init__(self):
        default_model_dir = 'src/myobjdet/myobjdet/model'
        default_model = 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
        default_labels = 'coco_labels.txt'

        path = os.getcwd()
        print(path)
        
        self.model = os.path.join(default_model_dir,default_model)
        self.labels = os.path.join(default_model_dir, default_labels)
        self.top_k = 1

        print('Loading {} with {} labels.'.format(self.model, self.labels))
        self.interpreter = make_interpreter(self.model)
        self.interpreter.allocate_tensors()
        self.labels = read_label_file(self.labels)
        self.inference_size = input_size(self.interpreter)
        self.threshold = 0.1

    def get_label_score(self, objs):
        if len(objs) >= 1:
            percent = int(100 * objs[0].score)
            label = '{}% {}'.format(percent, self.labels.get(objs[0].id, objs[0].id))
        else:
            percent = '--'
            label = '{}% {}'.format(percent, 'None')
        return label

    def detect(self, cv2_im):
        cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        cv2_im_rgb = cv2.resize(cv2_im_rgb, self.inference_size)
        run_inference(self.interpreter, cv2_im_rgb.tobytes())
        objs = get_objects(self.interpreter, self.threshold)[:self.top_k]

        return self.get_label_score(objs)


def main(args=None):
    rclpy.init(args=args)
    show_subscriber = ObjectDetector()
    rclpy.spin(show_subscriber)
    show_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
