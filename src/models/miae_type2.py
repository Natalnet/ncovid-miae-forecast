from models.miae import MIAE, RMSELoss
import time
import torch
from data_preparation import DataPreparation


class MIAET2(MIAE):
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
        self.model_type = "type2"

    def train(self, data_instance: DataPreparation, validation: bool = True):
        self.data_instance = data_instance
        self.data_train = data_instance.data_train
        if validation:
            self.data_validation = data_instance.data_test
        self.splitted_feature_data = data_instance.splited_data
        self.validation = validation

        epochs = range(self.epochs)
        start = time.time()

        self.train_all_autoencoders()
        self.set_decoders_weights_unadjustable()

        self.loss_train, self.loss_val = [], []
        for epoch in epochs:
            for batch in self.data_train:
                inputs, targets = batch
                forward_output = self.forward(inputs)

                loss = self.loss_calculation(batch, forward_output)
                self._weights_adjustment(loss)

            self.loss_train.append(loss.item())

            if self.validation:
                loss_validation = self.train_validation()
                self.loss_val.append(loss_validation.item())

        end = time.time()
        self.elapsed_time = end - start
        self.gen_score_to_save_model()

    def train_all_autoencoders(self):
        for idx, feature in enumerate(self.splitted_feature_data):
            autoencoder = [self.encoders[idx], self.decoders[idx]]
            self.train_autoencoder(feature, autoencoder)

    def train_autoencoder(self, feature, autoencoder):
        inner_optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        encoder = autoencoder[0]
        decoder = autoencoder[1]
        for epoch in range(self.epochs):
            for batch in feature:
                inputs, targets = batch
                encoded = encoder(inputs)
                decoded = decoder(encoded)

                loss = self.loss_function(decoded, targets)

                inner_optimizer.zero_grad()
                loss.backward()
                inner_optimizer.step()

    def train_validation(self):
        with torch.no_grad():
            for batch_validation in self.data_validation:
                inputs_validation, targets_validation = batch_validation

                forward_output = self.forward(inputs_validation)

                loss_validation = self.loss_validation_calculation(
                    batch_validation, forward_output
                )
        return loss_validation

    def loss_calculation(self, batch_validation, forward_output):
        inputs, targets = batch_validation
        decoded, predict = forward_output
        return self.loss_function(predict, targets)

    def loss_validation_calculation(self, batch_validation, forward_output):
        with torch.no_grad():
            inputs, targets = batch_validation
            decoded, predict = forward_output
            return self.loss_function(predict, targets)

    def set_decoders_weights_unadjustable(self):
        for param in self.decoders.parameters():
            param.requires_grad = False

    def set_encoders_weights_unadjustable(self):
        for param in self.encoders.parameters():
            param.requires_grad = False

    def set_autoencoders_weights_unadjustable(self):
        self.set_decoders_weights_unadjustable()
        self.set_encoders_weights_unadjustable()

    def gen_score_to_save_model(self):
        to_predict = self.data_instance.X_test
        pred = self.predicting(to_predict)

        ytrue = self.data_instance.Y_test
        yhat = pred

        self.score, self.scores = self.score_calculator(ytrue, yhat)
