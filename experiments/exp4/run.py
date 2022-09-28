import wandb
import sys
import torch
import numpy as np
from termcolor import cprint
from tqdm import tqdm
import yaml
from collections import namedtuple

sys.path.append('../../code')

from dataset_utils import *
from augmentations import *
from dataloader import MIdatasetMC
from model import *
from EEGNetContrastive import EEGNetContrastive, ClassifierOnTopOfPreTrainedEEGNet
from loss import *  # base_Model, base_Model_F, target_classifier
from constants import *


def EEGNet_finetune(
    train_dataset,
    test_dataset,
    epochs=50,
    mode=None,
    targetDsName=None,
    subjID=None,
    config=None,
):

    EEGnet = EEGNetContrastive(
        num_chan=config.num_chan,
        kerLen=config.kerLen,
        drop=config.drop,
        F1=config.F1,
        D=config.D,
        F2=config.F2,
        emdim=config.emdim,
        device=device,
    )
    if mode in [f'trained_unfrozen_fd', f'trained_frozen_fd']:
        cprint('initializing feature detector with pretrained weights', 'grey', 'on_red', attrs=['bold'])
        EEGnet.load_state_dict(torch.load(config.relPathToFDWeights))
    elif mode == f'untrained_unfrozen_fd':
        cprint('Starting with a clean feature detector', 'yellow', 'on_grey', attrs=['bold'])
    else:
        raise ValueError('Wrong mode')

    EEGnet.summary(num_time_samples=120)

    nclass = len(test_dataset['meta'].class_id.unique())
    cprint(f'number of classes: {nclass}', 'red')

    classifier = ClassifierOnTopOfPreTrainedEEGNet(
        emdim=config.emdim,
        num_classes=nclass,
    ).to(device)

    wandb.config = config._asdict()
    wandb.init(project=config.project_name,
               group=f'{mode}_{targetDsName}_{subjID}',
               entity=config.entity,
               config=wandb.config,
               save_code=False)
    wandb.watch((EEGnet, classifier), log_freq=10)

    feature_extractor_optimizer = torch.optim.AdamW(EEGnet.model.parameters(),
                                                    lr=config.lr,
                                                    betas=(config.betas[0], config.betas[1]),
                                                    weight_decay=config.weight_decay)

    classifer_criterion = nn.CrossEntropyLoss()
    classifier_optimizer = torch.optim.AdamW(classifier.parameters(),
                                             lr=config.lr,
                                             betas=(config.betas[0], config.betas[1]),
                                             weight_decay=config.weight_decay)

    # instantiate the Dataset class
    ds = MIdatasetMC(train_dataset, config)

    #     nt_xent_criterion = NTXentLoss_poly(
    #             device,
    #             configs.batch_size,
    #             configs.Context_Cont.temperature,
    #             configs.Context_Cont.use_cosine_similarity,
    #             batch_mult=1)  # needed for the reaarrange variant. Leave at default for others.

    # NOTE: The batch_size is the same as the length of the dataset itself
    nt_xent_criterion = CLIPLossX(device, len(ds))

    # instantiate a dataloader for the fine-tuning dataset
    # NOTE: The batch_size is the same as the length of the dataset itself
    dl = torch.utils.data.DataLoader(dataset=ds,
                                     batch_size=len(ds),
                                     shuffle=True,
                                     drop_last=True,
                                     num_workers=config.num_workers)  # optimal: 14 workers with batch_size=60

    EVAL_ACC = []
    evalAcc = 0
    for ep in range(epochs):
        epAcc = []
        pbar = tqdm(enumerate(dl))

        if mode in ['untrained_unfrozen_fd', 'trained_unfrozen_fd']:
            EEGnet.model.train()
        elif mode == 'trained_frozen_fd':
            EEGnet.model.eval()
        else:
            raise ValueError('Wrong mode')

        classifier.train()
        for i, (x_t, y, x_t_aug, x_f, x_f_aug, msk_t, msk_f, meta) in pbar:

            x_t = x_t.float().to(device)
            y = y.long().to(device)
            x_t_aug = x_t_aug.float().to(device)

            feature_extractor_optimizer.zero_grad()
            classifier_optimizer.zero_grad()

            # embed time representation and frequency representations (h). Z's are projections into joint TF space
            h_t = EEGnet.forward(x_t)  # original data
            h_t_aug = EEGnet.forward(x_t_aug)  # augmented data
            pred = classifier(h_t)

            class_loss = classifer_criterion(pred, y)  # predictor loss, actually, here is training loss
            feat_loss = nt_xent_criterion(h_t,
                                          h_t_aug)  # original and perturbed data should be close in TD embedding space

            if mode in ['untrained_unfrozen_fd', 'trained_unfrozen_fd']:
                total_loss = feat_loss + class_loss
            elif mode == 'trained_frozen_fd':
                total_loss = class_loss
            else:
                raise ValueError('Wrong mode')

            total_loss.backward()

            if mode in ['untrained_unfrozen_fd', 'trained_unfrozen_fd']:
                feature_extractor_optimizer.step()
                EEGnet.max_norm()
            classifier_optimizer.step()

            withinBatchAcc = np.mean([(p == l) for p, l in zip(pred.argmax(axis=1).tolist(), y.tolist())])
            epAcc.append(withinBatchAcc)
            pbar.set_description(
                f'epoch_acc: {np.mean(epAcc):.2f} evalAcc: {evalAcc:.2f} batch_acc: {withinBatchAcc:.2f}')

        # evaluate:
        EEGnet.model.eval()
        classifier.eval()
        x_t = test_dataset['samples'].float().to(device)
        y = test_dataset['labels']
        h_t = EEGnet.forward(x_t)
        pred = classifier(h_t)
        evalAcc = np.mean([(p == l) for p, l in zip(pred.argmax(axis=1).tolist(), y.tolist())])
        pbar.set_description(f'epoch_acc: {np.mean(epAcc):.2f} evalAcc: {evalAcc:.2f} batch_acc: {withinBatchAcc:.2f}')
        EVAL_ACC.append(evalAcc)

        performance_now = {
            'ep': ep,
            'feat_loss': feat_loss.item(),
            'class_loss': class_loss.item(),
            'total_loss': total_loss.item(),
            'train_acc': np.mean(epAcc),
            'eval_acc': evalAcc
        }
        wandb.log(performance_now)

    wandb.finish()
    return np.max(EVAL_ACC), np.mean(epAcc)


# load the config
yaml_txt = open('config.yaml').read()
config = yaml.load(yaml_txt, Loader=yaml.FullLoader)
config = namedtuple("config", config.keys())(*config.values())
modes = config.modes
targetDsName = config.targetDsName
test_subject = config.test_subject
relative_path_to_big_dataset = config.relative_path_to_big_dataset

# load and build data
shuffle = False
dataset = torch.load(relative_path_to_big_dataset)
pretrain_dataset, target_dataset = SplitIntoPretrainAndFinetune(dataset, targetDsName)

# run experiment (with one subject held out as the test subject)
for test_subject in target_dataset['meta'].subject.unique():
    train_dataset, test_dataset = get_train_test_AB(target_dataset, shuffle=shuffle, test_subject=test_subject)
    train_dataset = subset_train_dataset(train_dataset, subset_factor=5)
    # test_dataset, train_dataset = get_train_test_datasets(target_dataset, shuffle=shuffle, test_subject=test_subject)

    # run each mode three times
    for i in range(3):
        Nsubj = len(train_dataset['meta'].subject.unique())
        cprint(f"Number of subjects in train_dataset: {Nsubj}", color='blue')
        print(dataset['meta'].ds.unique())

        for mode in modes:
            EEGNet_finetune(train_dataset,
                            test_dataset,
                            epochs=config.epochs,
                            mode=mode,
                            targetDsName=targetDsName,
                            subjID=test_subject,
                            config=config)
