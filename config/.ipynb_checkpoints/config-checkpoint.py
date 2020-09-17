from easydict import EasyDict as edict

def get_split(sess, phone):
    return {'sess': sess, 'phone': phone}

config = edict()
config.combination = 5
# config.save_dir = f'ckpts/deeppix/oulu/corrected/proto3/{config.combination}'
config.save_dir = f'ckpts/deeppix/mobile-10/'
config.oulu_data_path = "/home/jupyter/datasets/spoof_data/cropped_faces/oulu_no_proto/NO_ALIGNMENT"
config.proto3 = edict()
config.proto4 = edict()

# protocol 3
config.proto3.train = [
    get_split([1, 2, 3], [1, 2, 3, 4, 5]),
    get_split([1, 2, 3], [1, 2, 3, 4, 6]),
    get_split([1, 2, 3], [1, 2, 3, 5, 6]),
    get_split([1, 2, 3], [1, 2, 4, 5, 6]),
    get_split([1, 2, 3], [1, 3, 4, 5, 6]),
    get_split([1, 2, 3], [2, 3, 4, 5, 6])
]

config.proto3.dev = [
    get_split([1, 2, 3], [1, 2, 3, 4, 5]),
    get_split([1, 2, 3], [1, 2, 3, 4, 6]),
    get_split([1, 2, 3], [1, 2, 3, 5, 6]),
    get_split([1, 2, 3], [1, 2, 4, 5, 6]),
    get_split([1, 2, 3], [1, 3, 4, 5, 6]),
    get_split([1, 2, 3], [2, 3, 4, 5, 6])
]

config.proto3.test = [
    get_split([1, 2, 3], [6]),
    get_split([1, 2, 3], [5]),
    get_split([1, 2, 3], [4]),
    get_split([1, 2, 3], [3]),
    get_split([1, 2, 3], [2]),
    get_split([1, 2, 3], [1])
]

# protocol 4
config.proto4.train = [
    get_split([1, 2], [1, 2, 3, 4, 5]),
    get_split([1, 2], [1, 2, 3, 4, 6]),
    get_split([1, 2], [1, 2, 3, 5, 6]),
    get_split([1, 2], [1, 2, 4, 5, 6]),
    get_split([1, 2], [1, 3, 4, 5, 6]),
    get_split([1, 2], [2, 3, 4, 5, 6])
]

config.proto4.dev = [
    get_split([1, 2], [1, 2, 3, 4, 5]),
    get_split([1, 2], [1, 2, 3, 4, 6]),
    get_split([1, 2], [1, 2, 3, 5, 6]),
    get_split([1, 2], [1, 2, 4, 5, 6]),
    get_split([1, 2], [1, 3, 4, 5, 6]),
    get_split([1, 2], [2, 3, 4, 5, 6])
]

config.proto4.test = [
    get_split([3], [6]),
    get_split([3], [5]),
    get_split([3], [4]),
    get_split([3], [3]),
    get_split([3], [2]),
    get_split([3], [1])
]
