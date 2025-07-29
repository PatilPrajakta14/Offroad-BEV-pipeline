import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import os, carla, cv2, numpy as np
from cv_bridge import CvBridge

class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')
        self.pub = self.create_publisher(Image, '/camera/image_raw', 10)
        self.bridge = CvBridge()

        host = os.getenv("CARLA_HOST", "localhost")
        client = carla.Client(host, 2000)

        client.set_timeout(10.0)
        world = client.get_world()
        bp    = world.get_blueprint_library()
        self.vehicle = world.spawn_actor(
            bp.filter('vehicle.*')[0],
            world.get_map().get_spawn_points()[0]
        )
        self.vehicle.set_autopilot(True)

        cam_bp = bp.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', '1280')
        cam_bp.set_attribute('image_size_y', '720')
        cam_bp.set_attribute('fov', '90')
        cam_tf = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.camera = world.spawn_actor(cam_bp, cam_tf, attach_to=self.vehicle)
        self.camera.listen(self._on_image)

    def _on_image(self, img):
        arr = np.frombuffer(img.raw_data, dtype=np.uint8)
        arr = arr.reshape((img.height, img.width, 4))[:, :, :3]
        cv_img = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        ros_img = self.bridge.cv2_to_imgmsg(cv_img, encoding='rgb8')
        self.pub.publish(ros_img)

    def destroy(self):
        self.camera.stop()
        self.camera.destroy()
        self.vehicle.destroy()

def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()