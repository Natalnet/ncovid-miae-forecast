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


class MIAE(nn.Module):
    def __init__(self, model_hyperparameters):
        super().__init__()
        self.model_hyperparameters = model_hyperparameters
        for item, value in model_hyperparameters.items():
            setattr(self, item, value)

        self.model_id = self.__gen_uiid()
        self.autoencoders_counter = 0

        if self.seed:
            self.fix_seed(self.seed)

        self.autoencoders_init()
        self.generate_autoencoders()
        self.generate_predictors()
        # have to be called after layer models
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.loss_function = RMSELoss()
        self.model_hyperparameters["optimizerFunction"] = "Adam"
        self.model_hyperparameters["lossFunction"] = "RMSE"

    def autoencoders_init(self):
        self.encoders, self.decoders, self.predictors = (
            nn.ModuleList(),
            nn.ModuleList(),
            nn.ModuleList(),
        )

    def fix_seed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

    def generate_autoencoders(self):
        for _ in range(self.n_features):
            self.add_autoencoder()

    def add_autoencoder(self):
        self.add_encoder()
        self.add_decoder()
        self.autoencoders_counter = self.autoencoders_counter + 1
        if self.is_an_appended_autoencoder():
            self.att_predictors()

    def is_an_appended_autoencoder(self):
        if self.n_features < self.autoencoders_counter:
            self.n_features = self.autoencoders_counter
            return True
        else:
            return False

    def add_encoder(self):
        self.encoders.append(
            nn.Sequential(
                nn.Linear(self.inseqlen, self.growth * self.inseqlen),
                getattr(nn, self.activation)(),
                nn.Linear(self.growth * self.inseqlen, self.latent_space_dim),
            )
        )

    def add_decoder(self):
        self.decoders.append(
            nn.Sequential(
                nn.Linear(self.latent_space_dim, self.growth * self.inseqlen),
                getattr(nn, self.activation)(),
                nn.Linear(self.growth * self.inseqlen, self.inseqlen),
            )
        )

    def generate_predictors(self):
        for _ in range(self.n_targets):
            self.add_predictor()

    def add_predictor(self):
        self.predictors.append(
            nn.Sequential(
                nn.Linear(self.n_features * self.latent_space_dim, self.outseqlen)
            )
        )

    def att_predictors(self):
        self.predictors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.n_features * self.latent_space_dim, self.outseqlen)
                )
                for _ in range(self.n_targets)
            ]
        )

    def forward(self, batch):
        encoded_batch, decoded_batch = self.forward_autoencoders(batch)
        predict_batch = self.forward_predictors(encoded_batch)

        return decoded_batch, predict_batch

    def forward_autoencoders(self, batch):
        encoded_batch, decoded_batch = [], []
        for input in batch:
            encoded_batch.append(
                [encoder(xs) for encoder, xs in zip(self.encoders, input)]
            )
        # stack is used to transform a list of tensor in a unic tensor of tensor
        for enc in encoded_batch:
            decoded_batch.append(
                torch.stack([decoder(z) for decoder, z in zip(self.decoders, enc)])
            )
        return encoded_batch, torch.stack(decoded_batch)

    def forward_predictors(self, encoded_batch):
        predict_batch = []
        # stack is used to transform a list of tensor in a unic tensor of tensor
        for enc in encoded_batch:
            predict_batch.append(
                torch.stack(
                    [predictor(torch.cat(enc, dim=-1)) for predictor in self.predictors]
                )
            )
        return torch.stack(predict_batch)

    def weights_adjustment(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

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
            "model_category": "miae",
            "model_type": self.model_type,
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

    def predicting(self, input_to_predict):
        with torch.no_grad():
            model_decoded, model_prediction = self.forward(input_to_predict)
        return model_prediction

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
