import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2, numpy as np

class VisualizationNode(Node):
    def __init__(self):
        super().__init__('visualization_node')
        self.bridge = CvBridge()
        self.bev_frame = None
        self.mask_frame = None
        self.create_subscription(Image, '/camera/bev_image', self.cb_bev, 10)
        self.create_subscription(Image, '/camera/bev_mask', self.cb_mask,10)

    def cb_bev(self, msg):
        self.bev_frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self._display()

    def cb_mask(self, msg):
        m = self.bridge.imgmsg_to_cv2(msg, 'mono8')
        self.mask_frame = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
        self._display()

    def _display(self):
        if self.bev_frame is None or self.mask_frame is None:
            return
        overlay = self.bev_frame.copy()
        overlay[self.mask_frame[:,:,0]>0] = (0,255,0)
        vis = cv2.addWeighted(overlay, 0.4, self.bev_frame, 1-0.4, 0)
        combined = np.hstack([self.bev_frame, self.mask_frame, vis])
        cv2.imshow("BEV | Mask | Overlay", combined)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = VisualizationNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()