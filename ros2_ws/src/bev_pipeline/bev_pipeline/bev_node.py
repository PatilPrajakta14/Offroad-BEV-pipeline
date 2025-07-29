import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2, numpy as np
from bev_pipeline.homography import compute_homography, apply_homography

class BEVNode(Node):
    def __init__(self):
        super().__init__('bev_node')
        self.sub = self.create_subscription(Image, '/camera/image_raw', self.cb, 10)
        self.pub = self.create_publisher(Image, '/camera/bev_image', 10)
        self.bridge = CvBridge()
        self.H = compute_homography()
        self.out_size = (512, 288)

    def cb(self, msg):
        cv_img = self.bridge.imgmsg_to_cv2(msg, 'rgb8')
        bev = apply_homography(cv_img, self.H, self.out_size)
        bev_msg = self.bridge.cv2_to_imgmsg(
            cv2.cvtColor(bev, cv2.COLOR_RGB2BGR), 'bgr8')
        self.pub.publish(bev_msg)

def main(args=None):
    rclpy.init(args=args)
    node = BEVNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()