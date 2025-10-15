class TConfig(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        del self[name]



POLICY_CONFIG = TConfig({
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
    "decoder_num": 7,
    'cam_keys': ['third_person_view', 'wrist_view'],
    'epoch_num': 45,
    "eps": 1e-9,
    "beta1": 0.9,
    "beta2": 0.98,
    "lr": 1e-5,
    "batch_size": 64
})
