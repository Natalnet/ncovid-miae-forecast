import sys

sys.path.append("../src")

import torch
import torch.nn as nn
from models.autoencoder import AutoEncoder, RMSELoss
from data_preparation import DataPreparation
import configures_manner

data_infos = {
    "repo": "p971074907",
    "path": "brl:rn",
    "inputFeatures": "date:newDeaths",
    "inputWindowSize": "7",
    "begin": "2020-03-13",
    "end": "2020-07-15",
}
configures_manner.add_all_configures_to_globals(data_infos)

repo = "p971074907"
path = "brl:rn"
inputFeatures = "date:newDeaths"
inputWindowSize = "7"
begin = "2020-03-13"
end = "2020-07-15"

data_instance = DataPreparation()
data = data_instance.get_data(repo, path, inputFeatures, inputWindowSize, begin, end)

forward_len = 7
data_instance.data_tensor_generate(forward_len)

prct_to_train = 0.7
data_instance.train_test_split_by_percent(prct_to_train)

batch_s = 8
data_instance.dataloader_create(batch_s)

model_hyperparameters = {
    "inseqlen": 7,
    "outseqlen": 7,
    "growth": 4,
    "latent_space_dim": 7,
    "n_features": 1,
    "n_targets": 1,
    "ae_archtecture_list": [20, 30, 50],
    "activation": "ReLU",
    "epochs": 100,
    "seed": 51,
    "learning_rate": 0.0005,
}

model = AutoEncoder(model_hyperparameters)
model.train(data_instance)
print("Model Trained")

to_predict = data_instance.X_test
pred = model.predicting(to_predict)

model.save_model()
ytrue = data_instance.Y_test
yhat = pred

s, ss = model.score_calculator(ytrue, yhat)
print(s, ss)
