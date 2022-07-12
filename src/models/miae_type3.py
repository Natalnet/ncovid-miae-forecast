from models.miae_type2 import MIAET2, RMSELoss
from data_preparation import DataPreparation


class MIAET3(MIAET2):
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
        self.model_type = "type3"

    def train(self, data_instance: DataPreparation, validation: bool = True):
        self.set_encoders_weights_unadjustable()
        MIAET2.train(self, data_instance, validation)
