import torch
import torch.nn as nn
import numpy as np
import random
import uuid
from sklearn.metrics import mean_squared_error
import json
from datetime import date, datetime
from data_preparation import DataPreparation
import configures_manner
import time


class AutoEncoder(nn.Module):
    def __init__(self, model_hyperparameters):
        super().__init__()
        self.model_hyperparameters = model_hyperparameters
        for item, value in model_hyperparameters.items():
            setattr(self, item, value)

        self.model_id = self.__gen_uiid()
        self.generate_autoencoder()

        # have to be called after layer models
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.loss_function = RMSELoss()
        self.model_hyperparameters["optimizerFunction"] = "Adam"
        self.model_hyperparameters["lossFunction"] = "RMSE"

    def fix_seed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

    def gen_encoder(self):
        self.encoder = nn.Sequential()

        enc_layer1 = nn.Linear(
            self.inseqlen * self.n_features, self.ae_archtecture_list[0]
        )

        self.encoder.append(enc_layer1)
        self.encoder.append(getattr(nn, self.activation)())

        for layer_index in range(len(self.ae_archtecture_list) - 1):
            self.encoder.append(
                nn.Linear(
                    self.ae_archtecture_list[layer_index],
                    self.ae_archtecture_list[layer_index + 1],
                )
            )
            self.encoder.append(getattr(nn, self.activation)())

    def gen_decoder(self):
        self.decoder = nn.Sequential()

        for layer_index in reversed(range(len(self.ae_archtecture_list))):
            self.decoder.append(
                nn.Linear(
                    self.ae_archtecture_list[layer_index],
                    self.ae_archtecture_list[layer_index - 1],
                )
            )
            self.decoder.append(getattr(nn, self.activation)())

        last_decoder_layer = nn.Linear(self.ae_archtecture_list[-1], self.outseqlen)
        self.decoder.append(last_decoder_layer)
        self.decoder.append(getattr(nn, self.activation)())

    def generate_autoencoder(self):
        self.gen_encoder()
        self.gen_decoder()

    def forward(self, batch):
        decoded_batch = []
        for input in batch:
            encoded = self.encoder(input)
            decoded = self.decoder(encoded)
            decoded_batch.append(decoded)
        return torch.stack(decoded_batch)

    def train(self, data_instance: DataPreparation, validation: bool = True) -> None:
        self.data_instance = data_instance
        self.data_train = data_instance.data_train
        self.validation = validation
        if self.validation:
            self.data_validation = data_instance.data_test

        epochs = range(self.epochs)

        self.loss_train, self.loss_val = [], []
        start = time.time()
        for epoch in epochs:
            for batch_train in self.data_train:
                inputs, targets = batch_train
                forward_output = self.forward(inputs)
                loss = self.loss_function(forward_output, targets)
                self._weights_adjustment(loss)

            self.loss_train.append(loss.item())

            if self.validation:
                loss_validation = self.train_validation()
                self.loss_val.append(loss_validation.item())

        end = time.time()
        self.elapsed_time = end - start
        self.gen_score_to_save_model()

    def _weights_adjustment(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train_validation(self):
        with torch.no_grad():
            for batch_validation in self.data_validation:
                inputs_validation, targets_validation = batch_validation

                forward_output = self.forward(inputs_validation)

                loss_validation = self.loss_function(forward_output, targets_validation)
        return loss_validation

    def predicting(self, input_to_predict):
        with torch.no_grad():
            model_prediction = self.forward(input_to_predict)
        return model_prediction

    def __gen_uiid(self):
        return str(uuid.uuid1())

    def save_model(self):
        model_path_to_save = configures_manner.model_path
        self.save_model_metadata_json()
        torch.save(self.parameters, model_path_to_save + self.model_id + ".pth")

    def _gen_model_metada(self):
        metadata_dict = {
            "instance_id": self.model_id,
            "score": self.score,
            "model_category": "autoencoder",
            "data_of_training": datetime.now().strftime("%Y-%m-%d " "%H:%M:%S"),
            "data_infos": {
                "path": configures_manner.path,
                "repo": configures_manner.repo,
                "data_begin_date": configures_manner.begin,
                "data_end_date": configures_manner.end,
                "inputFeatures": configures_manner.inputFeatures,
                "inputWindowSize": configures_manner.inputWindowSize,
            },
            "params": self.model_hyperparameters,
        }
        return metadata_dict

    def save_model_metadata_json(self):
        metadata_file_dict = self._gen_model_metada()
        with open(
            configures_manner.metadata_path + str(self.model_id) + ".json",
            "w",
        ) as fp:
            json.dump(metadata_file_dict, fp, indent=4)
        return metadata_file_dict

    def load_instace_model_from_id(self, model_id: str):
        model_path_to_load = configures_manner.model_path
        self.parameters = torch.load(model_path_to_load + model_id + ".pth")

    def gen_score_to_save_model(self):
        to_predict = self.data_instance.X_test
        pred = self.predicting(to_predict)

        ytrue = self.data_instance.Y_test
        yhat = pred

        self.score, self.scores = self.score_calculator(ytrue, yhat)

    def score_calculator(self, ytrue: torch.Tensor, ypred: torch.Tensor) -> tuple:
        scores = list()
        ytrue = ytrue.view(len(ytrue), self.outseqlen).numpy()
        ypred = ypred.view(len(ypred), self.outseqlen).numpy()

        for i in range(ytrue.shape[1]):
            mse = mean_squared_error(ytrue[:, i], ypred[:, i])
            rmse = np.sqrt(mse)
            scores.append(rmse)

        score_acumullator = 0
        for row in range(ytrue.shape[0]):
            for col in range(ytrue.shape[1]):
                score_acumullator += (ytrue[row, col] - ypred[row, col]) ** 2
        score = np.sqrt(score_acumullator / (ytrue.shape[0] * ytrue.shape[1]))

        return float(score), scores


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss
