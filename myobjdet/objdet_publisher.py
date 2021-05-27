import rclpy
from rclpy.node import Node

from std_msgs.msg import String

from detect import ObjDet


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

def main(args=None):
    rclpy.init(args=args)

    object_publisher = ObjDetPublisher()

    rclpy.spin(object_publisher)

    object_publisher.destroy_node()
    del object_publisher
    rclpy.shutdown()


if __name__ = '__main__':
    main()
