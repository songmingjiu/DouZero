import os

from douzero.dmc import parser, train

if __name__ == '__main__':
    # flags 保存了所有运行所需的参数
    flags = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = flags.gpu_devices
    train(flags)
