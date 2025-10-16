class TConfig(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        del self[name]


CONFIG = {
    'hardware_config': {
        'leader_config': {
            'port': '/dev/tty.usbmodem58FA0919711',
            'id': 'koval_los',
        },
        'follower_config': {
            'port': '/dev/tty.usbmodem58FA0930111',
            'id': 'koval_pes',

        },
        'cameras': {
            "wrist_view": {
                'index_or_path':0,
                'width': 640,
                'height': 480,
                'fps': 30
            },
            "third_person_view": {
                'index_or_path': 2,
                'width': 640,
                'height': 480,
                'fps': 30
            }
        }
    },
    'cam_keys': ['third_person_view', 'wrist_view'],
    'fps': 50,
    #'mean': [14.548566, -51.984997, 57.5864, 59.111923, 2.572637, 18.008852],
    #'std': [27.007866, 35.0966, 23.20677, 10.410391, 7.636047, 10.38727],
    'mean': [-0.10740731 ,-42.23653  ,   50.264027 ,   57.524323  ,  7.0488434, 25.104515],
    'std': [29.200237, 34.04495 , 25.577188 ,12.534714, 16.195923 ,14.92839],
    'robot_state_field': 'robot_state_follower', # robot_state_leader; whether to use follower or leader state as target
    'training_config': TConfig({
        "d_model": 512,
        "dropout": 0.1,
        "actions_dim": 6,
        "img_features": 1152,
        "h": 8,
        "prediction_horizon": 100,
        "H_patches": 16,
        "W_patches": 16,
        "d_internal": 1024,
        "encoder_num": 4,
        'cam_keys': ['third_person_view', 'wrist_view'],
        "decoder_num": 7,
        'epoch_num': 50,
        "eps": 1e-9,
        "beta1": 0.9,
        "beta2": 0.98,
        "lr": 1e-5,
        "batch_size": 64,
        "cam_keys": ['third_person_view', 'wrist_view']
    })
}
