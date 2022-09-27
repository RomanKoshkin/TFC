from pprint import pprint


class Config(object):

    def __init__(self):
        # model configs
        self.input_channels = 7
        self.increased_dim = 1
        self.final_out_channels = 128
        self.num_classes = 2
        self.num_classes_target = 2
        self.dropout = 0.35

        self.kernel_size = 25
        self.stride = 3

        self.features_len = 150
        self.features_len_f = 150
        self.TSlength_aligned = 150

        self.CNNoutput_channel = 10  # ???

        # training configs
        self.num_epoch = 40

        # optimizer parameters
        self.optimizer = 'adam'
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 3e-4  # 3e-4
        self.lr_f = self.lr

        # data parameters
        self.drop_last = True
        self.batch_size = 64
        """For Epilepsy, the target batchsize is 60"""
        self.target_batch_size = 60  # the size of target dataset (the # of samples used to fine-tune).

        self.Context_Cont = Context_Cont_configs()
        self.TC = TC()
        self.augmentation = augmentations()

    def report(self):
        pprint({i: self.__getattribute__(i) for i in dir(self) if not i.startswith('__')})
        print()
        pprint({i: self.augmentation.__getattribute__(i) for i in dir(self.augmentation) if not i.startswith('__')})


class augmentations(object):

    def __init__(self):

        # Frequency domain augmentation params
        self.pertubFreqRatio = 0.1  # how many % of the coefficients will be boosted
        self.boostFreqBy = 0.1  # by how many % of the chan-specific max FFT coeff do we boost chosen coefficients
        self.remove_how_much = 0.1  # how many % of the chang-spec FFT coeffs will be zeroed

        # Time domain augmentation params
        ## scaling
        self.scaling_mean = 2.0  # by how much on average the signal will be multiplied at each point
        self.scaling_sigma = 0.1  # by how much the multiplier will change

        ## jitter
        self.noise_std = 0.3  # the standard deviation of zero-mean additive white noise

        ## permutation
        self.max_seg = 12  # the number segments that each epoch will be split and shuffled along the time dimension

        ## whether we randomly choose one of the augmentations for each data point or mix them
        self.superposition = False


class Context_Cont_configs(object):

    def __init__(self):
        self.temperature = 0.2
        self.use_cosine_similarity = True


class TC(object):

    def __init__(self):
        self.hidden_dim = 64
        self.timesteps = 50