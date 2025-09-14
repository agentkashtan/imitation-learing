

class Follower:
    def __init__(self, robot, cameras):
        self.robot = robot
        self.cameras = cameras

    def send_action(self, action):
        self.robot.send_action(action)

    def get_observation(self):
        observation = { **self.robot.get_observation() }
        for cam_key, cam in self.cameras.items():
            observation[cam_key] = cam.get_frame()
        return observation

    def connect(self):
        self.robot.connect()

    def disconnect(self):
        self.robot.disconnect()