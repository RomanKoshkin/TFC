import torch
import random
import numpy as np
import pandas as pd
import os
import sys
import logging
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix, accuracy_score
from shutil import copy


def get_num_params(model):
    # get number of parameters
    a = 0
    for p in model.parameters():
        a += p.flatten().shape[0]
    print(f'number of parameters in model: {a}')


class HiddenPrints:

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


# compute the Lout for conv1d and maxpool1d layers in torch
def get_Lout(Lin=0, padding=0, dilation=0, kernel_sz=0, stride=0):
    return (Lin + 2 * padding - dilation * (kernel_sz - 1) - 1) // stride + 1


# compute the Lout of a conv+maxpool block
def getBlockOutLout(Lin=0, padding=0, dilation=0, kernel_sz=0, stride=0):
    # conv1d
    Lout = get_Lout(Lin=Lin, padding=padding, dilation=dilation, kernel_sz=kernel_sz, stride=stride)
    # MaxPool1d
    Lout = get_Lout(Lin=Lout, padding=1, dilation=1, kernel_sz=2, stride=2)
    return Lout


def set_requires_grad(model, dict_, requires_grad=True):
    for param in model.named_parameters():
        if param[0] in dict_:
            param[1].requires_grad = requires_grad


def fix_randomness(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def _calc_metrics(pred_labels, true_labels, log_dir, home_path):
    pred_labels = np.array(pred_labels).astype(int)
    true_labels = np.array(true_labels).astype(int)

    # save targets
    labels_save_path = os.path.join(log_dir, "labels")
    os.makedirs(labels_save_path, exist_ok=True)
    np.save(os.path.join(labels_save_path, "predicted_labels.npy"), pred_labels)
    np.save(os.path.join(labels_save_path, "true_labels.npy"), true_labels)

    r = classification_report(true_labels, pred_labels, digits=6, output_dict=True)
    cm = confusion_matrix(true_labels, pred_labels)
    df = pd.DataFrame(r)
    df["cohen"] = cohen_kappa_score(true_labels, pred_labels)
    df["accuracy"] = accuracy_score(true_labels, pred_labels)
    df = df * 100

    # save classification report
    exp_name = os.path.split(os.path.dirname(log_dir))[-1]
    training_mode = os.path.basename(log_dir)
    file_name = f"{exp_name}_{training_mode}_classification_report.xlsx"
    report_Save_path = os.path.join(home_path, log_dir, file_name)
    df.to_excel(report_Save_path)

    # save confusion matrix
    cm_file_name = f"{exp_name}_{training_mode}_confusion_matrix.torch"
    cm_Save_path = os.path.join(home_path, log_dir, cm_file_name)
    torch.save(cm, cm_Save_path)


def _logger(logger_name, level=logging.DEBUG):
    """
    Method to return a custom logger with the given name and level
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    # format_string = ("%(asctime)s — %(name)s — %(levelname)s — %(funcName)s:"
    #                 "%(lineno)d — %(message)s")
    format_string = "%(message)s"
    log_format = logging.Formatter(format_string)
    # Creating and adding the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    # Creating and adding the file handler
    file_handler = logging.FileHandler(logger_name, mode='a')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger


def copy_Files(destination, data_type):
    # destination: 'experiments_logs/Exp1/run1'
    destination_dir = os.path.join(destination, "model_files")
    os.makedirs(destination_dir, exist_ok=True)
    copy("main.py", os.path.join(destination_dir, "main.py"))
    copy("trainer.py", os.path.join(destination_dir, "trainer.py"))
    copy(f"../config_files/{data_type}_Configs.py", os.path.join(destination_dir, f"{data_type}_Configs.py"))
    copy("augmentations.py", os.path.join(destination_dir, "augmentations.py"))
    copy("dataloader.py", os.path.join(destination_dir, "dataloader.py"))
    copy(f"model.py", os.path.join(destination_dir, f"model.py"))
    copy("loss.py", os.path.join(destination_dir, "loss.py"))
    # copy("models/TC.py", os.path.join(destination_dir, "TC.py"))
