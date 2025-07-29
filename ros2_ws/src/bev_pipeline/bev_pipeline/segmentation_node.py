import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2, numpy as np
import torch
from bev_pipeline.cnn import UNetTiny

class SegmentationNode(Node):
    def __init__(self):
        super().__init__('segmentation_node')
        self.sub = self.create_subscription(Image, '/camera/bev_image', self.cb, 10)
        self.pub = self.create_publisher(Image, '/camera/bev_mask', 10)
        self.bridge = CvBridge()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # load the model from the package-relative path
        model_path = os.path.join(
            os.path.dirname(__file__),
            'models',
            'bev_model_best_val.pth'
        )
        self.model = UNetTiny().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def cb(self, msg):
        bgr = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        inp = torch.from_numpy(rgb.astype(np.float32)/255.0) \
                   .permute(2,0,1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(inp)[0,0].cpu().numpy()
        mask = (logits > 0.5).astype(np.uint8)*255
        mask_msg = self.bridge.cv2_to_imgmsg(mask, 'mono8')
        self.pub.publish(mask_msg)

def main(args=None):
    rclpy.init(args=args)
    node = SegmentationNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()