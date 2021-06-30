
# 项目配置文件
class Config():
    def __init__(self):

        self.data_root = './data/'
        self.data_test_root = './data/test/'
        self.log_path = './run/log/' + 'Alexnet'
        self.result_path = './result.csv'
        self.batch_size = 64
        self.epoch = 10


        self.lr = 1e-4
        self.input_size = 227
        # 通过抽样计算得到图片的均值mean和标准差std
        self.mean = [0.470, 0.431, 0.393]
        self.std = [0.274, 0.263, 0.260]