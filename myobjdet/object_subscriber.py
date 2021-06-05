import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class ObjectSubscriber(Node):
    def __init__(self):
        super().__init__("object_subscriber")
        self.create_subscription(String, "object_label", self.cb, 10)

    def cb(self, msg):
        self.get_logger().info("Detected: %s" % msg.data)

rclpy.init()
rclpy.spin(ObjectSubscriber())