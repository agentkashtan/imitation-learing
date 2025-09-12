from lerobot.robots.so101_follower import SO101FollowerConfig, SO101Follower
from lerobot.teleoperators.so101_leader import SO101LeaderConfig, SO101Leader
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig


CONFIG = {
    'leader_config': {
        'port': '/dev/tty.usbmodem58FA0919711',
        'id': 'koval_los',
    },
    'follower_config': {
        'port': '/dev/tty.usbmodem58FA0930111',
        'id': 'koval_pes',
        'cameras': {
            "third_person_view": OpenCVCameraConfig(index_or_path=0, width=1280, height=720, fps=30)
        }
    }
}


def get_follower():
    follower_config = SO101FollowerConfig(**CONFIG['follower_config'])
    return SO101Follower(follower_config)


def get_leader():
    leader_config = SO101LeaderConfig(**CONFIG['leader_config'])
    return SO101Leader(leader_config)
