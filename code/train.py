import imp
from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from dataloader import MIdatasetMC
from utils import HiddenPrints
from loss import NTXentLoss_poly
import wandb
from termcolor import cprint

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
    average_precision_score,
    accuracy_score,
    precision_score,
    f1_score,
    recall_score,
)
from sklearn.metrics import cohen_kappa_score as kappa


def get_loss_wts(a, b):
    w0 = b.abs().item() / (a.abs().item() + b.abs().item())
    w1 = a.abs().item() / (a.abs().item() + b.abs().item())
    return w0, w1


def finetune(feature_detector,
             classifier,
             finetune_dataset,
             test_dataset,
             configs,
             device,
             epochs,
             freeze=False,
             classifier_features='tf'):

    cprint('USING ONLY H_T FOR CLASSIFICATION', 'yellow', 'on_red', attrs=['bold'])

    wandb.config = configs
    wandb.init(project="TFC_1ch_rearrange_finetuning", entity="nightdude", config=wandb.config, save_code=False)
    wandb.watch((feature_detector, classifier), log_freq=10)

    # instantiate the Dataset class
    finetune_ds = MIdatasetMC(finetune_dataset, configs)

    # instantiate a dataloader for the fine-tuning dataset
    finetune_dl = torch.utils.data.DataLoader(dataset=finetune_ds,
                                              batch_size=configs.batch_size,
                                              shuffle=True,
                                              drop_last=True,
                                              num_workers=14)  # optimal: 14 workers with batch_size=60

    y = finetune_dataset['labels'].tolist()
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    for i, j in zip(np.unique(y), class_weights):
        print(f'\nClass {i} weight: {j.item():.2f}')

    model_optimizer = torch.optim.Adam(feature_detector.parameters(),
                                       lr=configs.lr,
                                       betas=(configs.beta1, configs.beta2),
                                       weight_decay=3e-4)

    classifier_optimizer = torch.optim.Adam(classifier.parameters(),
                                            lr=configs.lr,
                                            betas=(configs.beta1, configs.beta2),
                                            weight_decay=3e-4)

    # See explanation in TFC_7ch_Sandbox.ipynb
    nt_xent_criterion = NTXentLoss_poly(
        device,
        configs.batch_size,
        configs.Context_Cont.temperature,
        configs.Context_Cont.use_cosine_similarity,
        batch_mult=finetune_dataset['samples'].shape[1]
    )  # needed for the reaarrange variant. Leave at default for others.)  # device, 128, 0.2, True

    # reduction?
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights, reduction='mean')

    dat = []

    if freeze:
        for i, param in enumerate(feature_detector.parameters()):
            param.requires_grad = False
        cprint("Feature detector's parameters are frozen", color='blue')
    else:
        cprint("Feature detector's parameters are NOT frozen", color='blue')

    for ep in range(epochs):

        feature_detector.train()
        classifier.train()

        TCL = []
        FCL = []
        TFC = []
        TFaC = []
        TaFC = []
        TaFaC = []
        CL = []
        TotL = []
        Cacc = []
        AUC = []
        PRC = []
        REC = []
        PREC = []
        F1 = []
        FINETUNE_KAPPA = []

        pbar = tqdm(enumerate(finetune_dl))
        for i, (x_t, y, x_t_aug, x_f, x_f_aug, msk_t, msk_f, meta) in pbar:

            x_t, y = x_t.float().to(device), y.long().to(device)
            x_t_aug = x_t_aug.float().to(device)
            x_f, x_f_aug = x_f.float().to(device), x_f_aug.float().to(device)

            classifier_optimizer.zero_grad()

            # embed time representation and frequency representations (h). Z's are projections into joint TF space
            h_t, z_t, h_f, z_f = feature_detector(x_t, x_f)  # original data

            if (not freeze) or (ep < 2):
                model_optimizer.zero_grad()
                h_t_aug, z_t_aug, h_f_aug, z_f_aug = feature_detector(x_t_aug, x_f_aug)  # augmented data

                # contrastive time loss (encourages learning representations that are invariant to noise, translations and amplification)
                # contrastive frequency loss (encourages learning representations that are invariant to spectrum perturbations)
                loss_t = nt_xent_criterion(h_t,
                                           h_t_aug)  # original and perturbed data should be close in TD embedding space
                loss_f = nt_xent_criterion(h_f,
                                           h_f_aug)  # original and perturbed data should be close in FD embedding space

                # time and frequency embeddings should be similar when projected to the joint TF space
                l_TF, l_1, l_2, l_3 = (nt_xent_criterion(z_t, z_f), nt_xent_criterion(z_t, z_f_aug),
                                       nt_xent_criterion(z_t_aug, z_f), nt_xent_criterion(z_t_aug, z_f_aug))

                # Each of the 3 terms below optimize the model towards a smaller l_TF and relatively larger l_(1,2,3)
                # similar to triplet loss, l_TF should be smaller than l_{1,2,3}.
                # time-frequency consistency loss. same samples should be close in TF space, but augmented samples should not be so close to augmented/not augmented
                loss_c = (1 + l_TF - l_1) + (1 + l_TF - l_2) + (1 + l_TF - l_3)

            # classifier
            # fea_concat = torch.cat((z_t, z_f), dim=1)
            fea_concat = h_t

            predictions = classifier(fea_concat)  # how to define classifier? MLP? CNN?
            # fea_concat_flat = fea_concat.reshape(fea_concat.shape[0], -1)
            loss_class = criterion(predictions, y)  # predictor loss, actually, here is training loss

            if (not freeze) or (ep < 2):
                lam = 0.2
                feature_loss = (1 - lam) * loss_c + lam * (loss_t + loss_f)

                w0, w1 = get_loss_wts(loss_class, feature_loss)
                loss = w0 * loss_class + w1 * feature_loss
                pbar.set_description(f'L0: {loss_class:.2f} w0: {w0:.2f} L1: {feature_loss:.2f} w1: {w1:.2f}')
            else:
                loss = loss_class

            acc_bs = y.eq(predictions.detach().argmax(dim=1)).float().mean()
            onehot_label = F.one_hot(y, num_classes=2)
            pred_numpy = predictions.detach().cpu().numpy()

            auc_bs = roc_auc_score(onehot_label.detach().cpu().numpy(), pred_numpy, average="macro", multi_class="ovr")
            prc_bs = average_precision_score(onehot_label.detach().cpu().numpy(), pred_numpy)

            labels_numpy = y.detach().cpu().numpy()
            pred_numpy = np.argmax(pred_numpy, axis=1)

            FINETUNE_KAPPA.append(kappa(pred_numpy, labels_numpy))

            precision = precision_score(labels_numpy, pred_numpy, average="macro", zero_division=0)
            recall = recall_score(labels_numpy, pred_numpy, average="macro", zero_division=0)
            f1 = f1_score(labels_numpy, pred_numpy, average="macro", zero_division=0)

            TCL.append(loss_t.item() if not freeze else 0)
            FCL.append(loss_f.item() if not freeze else 0)
            TFC.append(l_TF.item() if not freeze else 0)
            TFaC.append(l_1.item() if not freeze else 0)
            TaFC.append(l_2.item() if not freeze else 0)
            TaFaC.append(l_3.item() if not freeze else 0)
            CL.append(loss_class.item())
            TotL.append(loss.item())
            Cacc.append(acc_bs.item())
            AUC.append(auc_bs)
            PRC.append(prc_bs)
            REC.append(recall)
            PREC.append(precision)
            F1.append(f1)

            loss.backward()
            if (not freeze) or (ep < 2):
                model_optimizer.step()
                pbar.set_description(f'L0: {loss_class:.2f} w0: {w0:.2f} L1: {feature_loss:.2f} w1: {w1:.2f}')
            classifier_optimizer.step()

        test_acc, test_kappa = evaluate(feature_detector, classifier, test_dataset, configs, device)
        performance_now = {
            'ep': ep,
            'temporal contrastive loss': np.mean(TCL),
            'frequency contrastive loss': np.mean(FCL),
            'time-frequency (t-f) consistency': np.mean(TFC),
            'time - aug. freq (t-~f) consistency': np.mean(TFaC),
            'aug.time - freq (~t-f) consistency': np.mean(TaFC),
            'aug.time - aug. freq (~t-~f) consistency': np.mean(TaFaC),
            'classifier loss': np.mean(CL),
            'total_loss': np.mean(TotL),
            'classifier accuracy': np.mean(Cacc),
            'AUC': np.mean(AUC),
            'PRC': np.mean(PRC),
            'kappa': np.mean(FINETUNE_KAPPA),
            'recall': np.mean(REC),
            'precision': np.mean(PREC),
            'F1-score': np.mean(F1),
            'test_acc': test_acc,
            'test_kappa': test_kappa,
        }

        dat.append(performance_now)
        wandb.log(performance_now)

    # either way
    for i, param in enumerate(feature_detector.parameters()):
        param.requires_grad = True
    cprint("Feature detector's parameters are unfrozen", color='blue')

    return pd.DataFrame(dat)


def evaluate(feature_detector, classifier, test_dataset, configs, device):
    feature_detector.eval()
    classifier.eval()

    P = []

    with HiddenPrints():
        test_ds = MIdatasetMC(test_dataset, configs)
        test_dl = torch.utils.data.DataLoader(
            dataset=test_ds,
            batch_size=1,
            shuffle=False,
            drop_last=True,
            num_workers=1,
        )

    for i, (x_t, y, x_t_aug, x_f, x_f_aug, msk_t, msk_f, meta) in enumerate(test_dl):

        x_t, y, x_f = x_t.float().to(device), y.long().to(device), x_f.float().to(device)

        # embed time representation and frequency representations (h). Z's are projections into joint TF space
        h_t, z_t, h_f, z_f = feature_detector(x_t, x_f)  # original data

        # classifier
        # fea_concat = torch.cat((z_t, z_f), dim=1)
        fea_concat = h_t
        predictions = classifier(fea_concat)  # how to define classifier? MLP? CNN?

        pred_numpy = predictions.detach().cpu().numpy()
        labels_numpy = y.detach().cpu().numpy()

        P.append((np.argmax(pred_numpy[0]), y.item()))

    accuracy = np.mean([i[0] == i[1] for i in P])
    k = kappa(np.array(P)[:, 0], np.array(P)[:, 1])
    return accuracy, k
