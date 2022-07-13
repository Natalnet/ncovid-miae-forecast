import sys

sys.path.append("../src")

from data_preparation import DataPreparation
from models.miae_type3 import MIAET3, RMSELoss
import matplotlib.pyplot as plt
import torch

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
data_instance.train_test_split(prct_to_train)

batch_s = 8
data_instance.dataloader_create(batch_s)
data_instance.data_split_by_feature()

features = data_instance.input_features
targets_features = [data_instance.targets_features]
# Window lenght for the input data
input_window = data_instance.window_size
# Desire prediction output lenght

learning_rate = 0.0005

model_hyperparameters = {
    "inseqlen": input_window,
    "outseqlen": forward_len,
    "growth": 4,
    "latent_space_dim": 7,
    "n_features": len(features),
    "n_targets": len(targets_features),
    "activation": "ReLU",
    "epochs": 100,
    "seed": 51,
    "learning_rate": 0.0005,
    "loss_function": RMSELoss(),
}

model = MIAET3(model_hyperparameters)

model.train(data_instance)
print("Model Trained")

to_predict = data_instance.X_test
pred = model.predicting(to_predict)

ytrue = data_instance.Y_test
yhat = pred

s, ss = model.score_calculator(ytrue, yhat)
print(s, ss)
