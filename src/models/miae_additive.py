from models.miae import MIAE, RMSELoss
import time
import torch
from data_preparation import DataManagement


class MIAET1(MIAE):
    def __init__(self, atribucts_dict):
        super().__init__(atribucts_dict)

        for item, value in atribucts_dict.items():
            setattr(self, item, value)

        self.autoencoders_counter = 0

        if self.seed:
            self.fix_seed(self.seed)

        self.autoencoders_init()
        self.generate_autoencoders()
        self.generate_predictors()

    def train_additive(self, new_feature):
        autoencoder = [self.encoders[-1], self.decoders[-1]]
        self.train_autoencoder(new_feature, autoencoder)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )
        self.train_predictor()

    def train_predictor(self):
        epochs = range(self.epochs)

        start = time.time()
        self.loss_train, self.loss_val = [], []
        for epoch in epochs:
            for batch in self.data_train:
                inputs, targets = batch
                forward_output = self.forward(inputs)

                loss = self.splitted_loss_calculation(batch, forward_output)
                self.weights_adjustment(loss)

            self.loss_train.append(loss.item())
        end = time.time()
        self.elapsed_time = end - start
