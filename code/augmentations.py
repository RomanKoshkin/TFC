import numpy as np
import torch
import warnings
import torch.nn.functional as F

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


# https://arxiv.org/pdf/1706.00527.pdf
def jitter(x, sigma=0.8):
    # mix in some white noise
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0.0, scale=sigma, size=x.shape)


# https://arxiv.org/pdf/1706.00527.pdf
def scaling(x, mu=2.0, sigma=1.1):
    # make voltage deflections bigger. Don't overdo with sigma, because that'd make it similar to jitter
    factor = torch.randn(size=x.shape) * sigma + mu
    return x * factor


# @torch.jit.script
def permutation(x, max_segments=5):
    if len(x.shape) == 2:
        x = x.unsqueeze(0)
    elif len(x.shape) == 3:
        pass
    else:
        raise ValueError('Wrong tensor shape. Must be either 2 or 3-d.')
    orig_steps = np.arange(x.shape[2])
    for i in range(x.shape[0]):
        # get a unique vector of sorting indices for each epoch:
        num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))
        split_points = np.random.choice(x.shape[2] - 2, num_segs[i] - 1, replace=False)
        split_points.sort()
        # random permutation on arrays of diff size gives a warning, but works 25 faster than if you got rid of this warning and used a hack with lists
        splits = np.split(orig_steps, split_points)
        sorting_indices = np.concatenate(np.random.permutation(splits))
        for ch in range(x.shape[1]):
            # sort each channel in an epoch in the SAME way
            x[i, ch, :] = x[i, ch, sorting_indices]
    return x.squeeze()


def remove_frequency(x, remove_how_much=0.1):
    mask = torch.FloatTensor(x.shape).uniform_() > remove_how_much  # maskout_ratio are False
    return x * mask


def add_frequency(x, pertubFreqRatio=0, boostFreqBy=0.1):
    # Boolean mask. Only pertub_ratio of all values are True
    mask = torch.FloatTensor(x.shape).uniform_() < pertubFreqRatio
    # max_amplitude = x.max() # if you have one-channel data

    # compute channel-specific FFT coefficients' maxima, along the 'samples' dim (not batch_dim, not channels)
    max_amplitude = x.max(axis=-1, keepdim=True)[0]

    # create a matrix of random numbers in the range [0, 0.1*channel-specific max FFT coefficient]
    random_am = torch.rand(mask.shape) * (max_amplitude * boostFreqBy)

    # make this matrix sparse.
    pertub_matrix = mask * random_am

    # based on that sparse matrix, INCREASE selected FFT coefficients
    return x + pertub_matrix


def one_hot_encoding(X):
    X = [int(x) for x in X]
    n_values = np.max(X) + 1
    b = np.eye(n_values)[X]
    return b


def DataTransform_TD(sample, config):
    aug_1 = jitter(sample.clone(), sigma=config.augmentation.noise_std)
    aug_2 = scaling(sample.clone(), mu=config.augmentation.scaling_mean, sigma=config.augmentation.scaling_sigma)
    aug_3 = permutation(sample.clone(), max_segments=config.augmentation.max_seg)
    if len(sample.shape) == 3:
        msk = torch.rand(size=(aug_1.shape[0], 3)).max(axis=1)[1]
        msk = F.one_hot(msk, num_classes=3)
        aug_T = aug_1 * msk[:, 0].reshape(-1, 1, 1) + aug_2 * msk[:, 1].reshape(-1, 1, 1) + aug_3 * msk[:, 2].reshape(
            -1, 1, 1)
        return aug_T, aug_1, aug_2, aug_3, msk.squeeze()
    elif len(sample.shape) == 2:
        msk = torch.randint(high=3, size=(1,))
        msk = F.one_hot(msk, num_classes=3)

        # now its just one of many, but you can make a mixture with softmax
        aug_T = aug_1 * msk[:, 0] + aug_2 * msk[:, 1] + aug_3 * msk[:, 2]
        return aug_T, aug_1, aug_2, aug_3, msk.squeeze()
        # msk = torch.randint(high=3, size=(1,))
        # return (aug_1, aug_2, aug_3)[msk], aug_1, aug_2, aug_3, msk
    else:
        raise ValueError('Wrong tensor shape. Must be either 2 or 3-d.')


def DataTransform_FD(sample, config):
    aug_1 = remove_frequency(sample.clone(), remove_how_much=config.augmentation.remove_how_much)
    aug_2 = add_frequency(sample.clone(),
                          pertubFreqRatio=config.augmentation.pertubFreqRatio,
                          boostFreqBy=config.augmentation.boostFreqBy)

    if len(sample.shape) == 3:
        msk = torch.rand(size=(aug_1.shape[0], 2)).max(axis=1)[1]
        msk = F.one_hot(msk, num_classes=2).unsqueeze(1).unsqueeze(-1)
        aug_F = aug_1 * msk[:, :, 0] + aug_2 * msk[:, :, 1]
        return aug_F, aug_1, aug_2, msk.squeeze()
    elif len(sample.shape) == 2:
        if config.augmentation.superposition:
            msk = torch.rand(size=(1, 2))
            msk = F.softmax(msk, dim=1).unsqueeze(1)
        else:
            msk = torch.rand(size=(1, 2)).max(axis=1)[1]
            msk = F.one_hot(msk, num_classes=2)
        aug_F = aug_1 * msk[:, 0] + aug_2 * msk[:, 1]
        return aug_F, aug_1, aug_2, msk.squeeze()


# def DataTransform_TD(sample, config):
#     """Weak and strong augmentations"""
#     aug_1 = jitter(sample, config.augmentation.jitter_ratio)
#     aug_2 = scaling(sample, config.augmentation.jitter_scale_ratio)
#     aug_3 = permutation(sample, max_segments=config.augmentation.max_seg)

#     li = np.random.randint(0, 4, size=[sample.shape[0]])  # there are two augmentations in Frequency domain
#     li_onehot = one_hot_encoding(li)
#     aug_1[1 - li_onehot[:, 0]] = 0  # the rows are not selected are set as zero.
#     aug_2[1 - li_onehot[:, 1]] = 0
#     aug_3[1 - li_onehot[:, 2]] = 0
#     # aug_4[1 - li_onehot[:, 3]] = 0
#     aug_T = aug_1 + aug_2 + aug_3  #+aug_4
#     return aug_T

# def DataTransform_FD(sample, config):
#     """Weak and strong augmentations in Frequency domain """
#     aug_1 = remove_frequency(sample, 0.1)
#     aug_2 = add_frequency(sample, 0.1)
#     # generate random sequence
#     li = np.random.randint(0, 2, size=[sample.shape[0]])  # there are two augmentations in Frequency domain
#     li_onehot = one_hot_encoding(li)
#     aug_1[1 - li_onehot[:, 0]] = 0  # the rows are not selected are set as zero.
#     aug_2[1 - li_onehot[:, 1]] = 0
#     aug_F = aug_1 + aug_2
#     return aug_F