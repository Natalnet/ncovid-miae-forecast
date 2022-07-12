from models.miae import MIAE, RMSELoss
import time
import torch
from data_preparation import DataPreparation


class MIAET1(MIAE):
    def __init__(self, model_hyperparameters):
        super().__init__(model_hyperparameters)

        for item, value in model_hyperparameters.items():
            setattr(self, item, value)

        self.autoencoders_counter = 0

        if self.seed:
            self.fix_seed(self.seed)

        self.autoencoders_init()
        self.generate_autoencoders()
        self.generate_predictors()
        self.model_type = "type1"

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

                loss = self.loss_calculation(batch_train, forward_output)
                self._weights_adjustment(loss)

            self.loss_train.append(loss.item())

            if self.validation:
                loss_validation = self.train_validation()
                self.loss_val.append(loss_validation.item())

        end = time.time()
        self.elapsed_time = end - start
        self.gen_score_to_save_model()

    def train_validation(self):
        with torch.no_grad():
            for batch_validation in self.data_validation:
                inputs_validation, targets_validation = batch_validation

                forward_output = self.forward(inputs_validation)

                joined_loss_validation = self.loss_validation_calculation(
                    batch_validation, forward_output
                )
        return joined_loss_validation

    def loss_calculation(self, batch_train, forward_output):
        inputs, targets = batch_train
        decoded, predict = forward_output
        autoencoder_loss = self.loss_function(decoded, inputs)
        predicted_loss = self.loss_function(predict, targets)
        return autoencoder_loss + predicted_loss

    def loss_validation_calculation(self, batch_validation, forward_output):
        with torch.no_grad():
            inputs, targets = batch_validation
            decoded, predict = forward_output
            autoencoder_loss = self.loss_function(decoded, inputs)
            predicted_loss = self.loss_function(predict, targets)
            return autoencoder_loss + predicted_loss

    def gen_score_to_save_model(self):
        to_predict = self.data_instance.X_test
        pred = self.predicting(to_predict)

        ytrue = self.data_instance.Y_test
        yhat = pred

        self.score, self.scores = self.score_calculator(ytrue, yhat)
