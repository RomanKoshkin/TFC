from re import T
from sklearn.model_selection import train_test_split, KFold
import torch, copy
from termcolor import cprint
import numpy as np


def SplitIntoPretrainAndFinetune(_dataset, finetune_dsname):
    ''' 
    Split the BIG dataset in to pretrain and finetune, dataset-wise.
    '''

    cprint(f'{finetune_dsname}', color='green')

    pretrain_dataset = copy.deepcopy(_dataset)
    idx = np.where(pretrain_dataset['meta'].ds != finetune_dsname)[0]  # get lines NOT corresponding to the selected ds
    pretrain_dataset['meta'] = pretrain_dataset['meta'].iloc[idx]
    pretrain_dataset['labels'] = pretrain_dataset['labels'][idx]
    pretrain_dataset['samples'] = pretrain_dataset['samples'][idx, :, :]
    cprint(f"pretrain_dataset: {pretrain_dataset['labels'].shape}", color='grey')

    target_dataset = copy.deepcopy(_dataset)  # make a copy of the entire big dataset
    idx = np.where(target_dataset['meta'].ds == finetune_dsname)[0]  # get lines corresponding to the selected ds
    target_dataset['meta'] = target_dataset['meta'].iloc[idx]  #
    target_dataset['labels'] = torch.tensor(target_dataset['meta'].class_id.tolist()).long()
    target_dataset['samples'] = target_dataset['samples'][idx, :, :]
    cprint(target_dataset['meta']['label'].unique(), color='yellow')

    if finetune_dsname == 'BNCISchirrmeister2017.pt':
        d = {'right_hand': 0, 'left_hand': 1, 'rest': 2, 'feet': 3}
    elif finetune_dsname == 'BNCIWeibo2014.pt':
        d = {
            'right_hand': 0,
            'left_hand': 1,
            'rest': 2,
            'feet': 3,
            'hands': 4,
            'left_hand_right_foot': 5,
            'right_hand_left_foot': 6
        }
    elif finetune_dsname == 'BNCIAlexandreMotorImagery.pt':
        d = {'right_hand': 0, 'feet': 1, 'rest': 2}
    elif finetune_dsname == 'BNCIOfner2017.pt':
        d = {'right_hand_close': 0, 'right_hand_open': 1, 'right_pronation': 2, 'right_supination': 3}
    elif finetune_dsname == 'BNCIPhysionetMotorImagery.pt':
        d = {'right_hand': 0, 'left_hand': 1}
    elif finetune_dsname == 'BNCIGrosse-Wentrup2009.pt':
        d = {'right_hand': 0, 'left_hand': 1}
    elif finetune_dsname == 'BNCIShin2017A.pt':
        d = {'right_hand': 0, 'left_hand': 1}
    elif finetune_dsname == 'BNCIZhou2016.pt':
        d = {'right_hand': 0, 'left_hand': 1}
    elif finetune_dsname == 'BNCICho2017.pt':
        d = {'right_hand': 0, 'left_hand': 1}
    elif finetune_dsname == 'BNCI001-2014.pt':
        d = {'right_hand': 0, 'left_hand': 1}
    elif finetune_dsname == 'BNCI001-2015.pt':
        d = {'right_hand': 0, 'feet': 1}
    elif finetune_dsname == 'BNCI004-2015.pt':
        d = {'right_hand': 0, 'feet': 1}
    else:
        d = {i: j for j, i in enumerate(target_dataset['meta'].label.unique())}  # build a map
    # map class labels to numbers from 0 onwards
    target_dataset['meta']['class_id'] = target_dataset['meta'].label.map(d)
    idx = np.where(target_dataset['meta'].class_id.isin([0, 1]))[0]  # where class labels are 0 and 1
    target_dataset['meta'] = target_dataset['meta'].iloc[idx]  # leave only the entries with class lables 0 and 1
    target_dataset['labels'] = torch.tensor(target_dataset['meta'].class_id.tolist()).long()
    target_dataset['samples'] = target_dataset['samples'][idx, :, :]
    cprint(f"target_dataset: {target_dataset['labels'].shape}", color='grey')
    print(
        f"Total samples in pretrain and finetune: {pretrain_dataset['labels'].shape[0] + target_dataset['labels'].shape[0]}"
    )
    return pretrain_dataset, target_dataset


def get_train_test_datasets(target_dataset, shuffle=False, test_subject=None):
    '''
    split the target dataset (which is supposed to be from one experiement) 
    into train (one one which you'll fine-tune) and test which for now we'll use
    for validation during the fine-tuning
    '''

    train_dataset = copy.deepcopy(target_dataset)
    test_dataset = copy.deepcopy(target_dataset)

    if not test_subject:
        cprint(f'No subject provided. Defaulting to 80/20 split.', 'yellow', 'on_green')
        (train_idx, test_idx) = train_test_split(
            list(range(target_dataset['labels'].shape[0])),
            test_size=0.2,
            shuffle=shuffle,
        )
    else:
        cprint(f'test_subject: {test_subject}', 'yellow')
        train_idx = np.where(target_dataset['meta'].subject != test_subject)[0]
        test_idx = np.where(target_dataset['meta'].subject == test_subject)[0]

    train_dataset['samples'] = train_dataset['samples'][train_idx, :, :]
    train_dataset['labels'] = train_dataset['labels'][train_idx]
    train_dataset['meta'] = train_dataset['meta'].iloc[train_idx]

    test_dataset['samples'] = test_dataset['samples'][test_idx, :, :]
    test_dataset['labels'] = test_dataset['labels'][test_idx]
    test_dataset['meta'] = test_dataset['meta'].iloc[test_idx]

    cprint(f"Shuffle: {shuffle}", color='red', attrs=['bold'])
    cprint(f"{train_dataset['samples'].shape}, {test_dataset['samples'].shape}", color='yellow')

    return train_dataset, test_dataset


def get_train_test_AB(target_dataset, shuffle=False, test_subject=None):
    '''
    split the target dataset (which is supposed to be from one experiement) 
    into train (one one which you'll fine-tune) and test which for now we'll use
    for validation during the fine-tuning
    '''

    if not test_subject:
        raise ValueError('Provide test subject ID')
    else:
        cprint(f'test_subject: {test_subject}', 'yellow')

    # now select session A (for training) and session B (for testing) from the test_dataset
    sess_IDs = target_dataset['meta'].session.unique().tolist()

    train_idx = np.where((target_dataset['meta'].session == sess_IDs[0]) &
                         (target_dataset['meta'].subject == test_subject))[0]
    test_idx = np.where((target_dataset['meta'].session == sess_IDs[1]) &
                        (target_dataset['meta'].subject == test_subject))[0]

    train_dataset = copy.deepcopy(target_dataset)
    test_dataset = copy.deepcopy(target_dataset)

    train_dataset['samples'] = train_dataset['samples'][train_idx, :, :]
    train_dataset['labels'] = train_dataset['labels'][train_idx]
    train_dataset['meta'] = train_dataset['meta'].iloc[train_idx]

    test_dataset['samples'] = test_dataset['samples'][test_idx, :, :]
    test_dataset['labels'] = test_dataset['labels'][test_idx]
    test_dataset['meta'] = test_dataset['meta'].iloc[test_idx]

    cprint(f"Shuffle: {shuffle}", color='red', attrs=['bold'])
    cprint(f"{train_dataset['samples'].shape}, {test_dataset['samples'].shape}", color='yellow')

    return train_dataset, test_dataset


def subset_train_dataset(train_dataset, subset_factor=5):
    right_idx = np.where(train_dataset['meta'].label == 'right_hand')[0]
    left_idx = np.where(train_dataset['meta'].label == 'left_hand')[0]

    left_idx = np.random.choice(left_idx, len(left_idx) // subset_factor, replace=False)
    right_idx = np.random.choice(right_idx, len(right_idx) // subset_factor, replace=False)
    subset = np.random.permutation(np.concatenate([right_idx, left_idx]))

    train_dataset['meta'] = train_dataset['meta'].iloc[subset]
    train_dataset['labels'] = train_dataset['labels'][subset]
    train_dataset['samples'] = train_dataset['samples'][subset, :, :]
    return train_dataset


class TrainTestSplitKfold(object):
    """ Iterabale object that return CV folds over the target dataset """

    def __init__(self, folds=5, target_dataset=None):
        self.ds = copy.deepcopy(target_dataset)
        self.kf = KFold(n_splits=folds)
        self.length = self.kf.get_n_splits(target_dataset['labels'])
        self.g = self.kf.split(target_dataset['meta'])
        cprint(self.length, color='yellow')

    def __len__(self):
        return self.length

    def __iter__(self):
        return self

    def __next__(self):
        train_index, test_index = next(self.g)

        train_meta = self.ds['meta'].iloc[train_index]
        train_labels = self.ds['labels'][train_index]
        train_samples = self.ds['samples'][train_index, :, :]

        test_meta = self.ds['meta'].iloc[test_index]
        test_labels = self.ds['labels'][test_index]
        test_samples = self.ds['samples'][test_index, :, :]

        train_dataset = {
            'samples': train_samples,
            'labels': train_labels,
            'meta': train_meta,
        }
        test_dataset = {
            'samples': test_samples,
            'labels': test_labels,
            'meta': test_meta,
        }

        return train_dataset, test_dataset
