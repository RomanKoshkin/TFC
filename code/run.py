import torch
import os, sys

# path hacks
os.chdir("code")
sys.path.insert(1, os.path.join(sys.path[0], ".."))
import numpy as np
from datetime import datetime
import argparse
from utils import _logger, set_requires_grad
from model import *
from utils import _calc_metrics, copy_Files
from loss import *  # base_Model, base_Model_F, target_classifier
from datetime import datetime

from dataloader import data_generator
from trainer import Trainer, model_finetune, model_test  # model_evaluate

print(f"CUDA available: {torch.cuda.is_available()}")

device_count = torch.cuda.device_count()
for i in range(device_count):
    torch.cuda.set_device(i)
    cur_device_id = torch.cuda.current_device()
    cur_device_name = torch.cuda.get_device_name(cur_device_id)
    print(f"Current device:\nID {cur_device_id} | Name: {cur_device_name}")
    print(f"supported arch list: {torch.cuda.get_arch_list()}\n")

run_description = "run1"
SEED = 0
training_mode = "pre_train"  # 'fine_tune_test'
target_dataset = "Epilepsy"  # 'FD_B', 'Gesture', 'EMG'
sourcedata = "SleepEEG"  # 'FD_A', 'HAR', 'ECG'
targetdata = "Epilepsy"  # 'Epilepsy, 'FD_B', 'Gesture', 'EMG'
logs_save_dir = "experiments_logs"
device = "cuda:1"
experiment_description = str(sourcedata) + "_2_" + str(targetdata)
method = "Time-Freq Consistency"  # 'TS-TCC'

home_dir = os.getcwd()
home_path = home_dir
os.makedirs(logs_save_dir, exist_ok=True)

exec(f"from config_files.{sourcedata}_Configs import Config as Configs")
configs = Configs()  # THis is OK???

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

experiment_log_dir = os.path.join(
    logs_save_dir,
    experiment_description,
    run_description,
    str(training_mode) + f"_seed_{SEED}",
)
os.makedirs(experiment_log_dir, exist_ok=True)

print(experiment_log_dir)

# loop through domains
counter = 0
src_counter = 0

start_time = datetime.now()
# Logging
log_file_name = os.path.join(experiment_log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
# 'experiments_logs/Exp1/run1/train_linear_seed_0/logs_14_04_2022_15_13_12.log'
logger = _logger(log_file_name)
logger.debug("=" * 45)
logger.debug(f"Pre-training Dataset: {sourcedata}")
logger.debug(f"Target (fine-tuning) Dataset: {targetdata}")
logger.debug(f"Method:  {method}")
logger.debug(f"Mode:    {training_mode}")
logger.debug("=" * 45)

# Load datasets
sourcedata_path = f"../datasets/{sourcedata}"  # './data/Epilepsy'
targetdata_path = f"../datasets/{targetdata}"
# for self-supervised, the data are augmented here. Only self-supervised learning need augmentation
subset = False  # if subset= true, use a subset for debugging.
train_dl, valid_dl, test_dl = data_generator(sourcedata_path, targetdata_path, configs, training_mode, subset=subset)
logger.debug("Data loaded ...")

# Load Model
"""Here are two models, one basemodel, another is temporal contrastive model"""
# model = Time_Model(configs).to(device)
# model_F = Frequency_Model(configs).to(device) #base_Model_F(configs).to(device) """here is right. No bug in this line.
TFC_model = TFC(configs).to(device)
classifier = target_classifier(configs).to(device)

temporal_contr_model = None  # TC(configs, device).to(device)

model_optimizer = torch.optim.Adam(
    TFC_model.parameters(),
    lr=configs.lr,
    betas=(configs.beta1, configs.beta2),
    weight_decay=3e-4,
)
classifier_optimizer = torch.optim.Adam(
    classifier.parameters(),
    lr=configs.lr,
    betas=(configs.beta1, configs.beta2),
    weight_decay=3e-4,
)
temporal_contr_optimizer = None  # torch.optim.Adam(temporal_contr_model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)

if training_mode == "pre_train":  # to do it only once
    copy_Files(os.path.join(logs_save_dir, experiment_description, run_description), sourcedata)

# Trainer
Trainer(
    TFC_model,
    temporal_contr_model,  # None
    model_optimizer,
    temporal_contr_optimizer,  # None
    train_dl,
    valid_dl,
    test_dl,
    device,
    logger,
    configs,
    experiment_log_dir,
    training_mode,
    model_F=None,
    model_F_optimizer=None,
    classifier=classifier,
    classifier_optimizer=classifier_optimizer,
)

logger.debug(f"Training time is : {datetime.now()-start_time}")
