from data_preparation import DataPreparation
from models.miae_type1 import MIAET1, RMSELoss
from models.miae_type2 import MIAET2
from models.miae_type3 import MIAET3
import configures_manner


class Evaluator:
    def __init__(self):
        super().__init__()
        self.gen_instace_to_train()
        self.gen_data_to_train()

    def generate_type1_instance(self):
        model_instance = MIAET1(configures_manner.modelConfigs)
        return model_instance

    def generate_type2_instance(self):
        model_instance = MIAET2(configures_manner.modelConfigs)
        return model_instance

    def generate_type3_instance(self):
        model_instance = MIAET3(configures_manner.modelConfigs)
        return model_instance

    def gen_instace_to_train(self):
        self.model = getattr(
            self, f"generate_{configures_manner.model_type}_instance"
        )()

    def train(self):
        self.model.train(self.data_instance)

    def gen_data_to_train(self):

        data_instance = DataPreparation()
        data_instance.get_data(
            configures_manner.repo,
            configures_manner.path,
            configures_manner.inputFeatures,
            configures_manner.inputWindowSize,
            configures_manner.begin,
            configures_manner.end,
        )
        data_instance.outseqlen = self.model.outseqlen
        data_instance.inseqlen = self.model.inseqlen

        data_instance.data_tensor_generate(configures_manner.outputWindowSize)
        data_instance.train_test_split_by_days(configures_manner.testSize)
        data_instance.dataloader_create()
        data_instance.data_split_by_feature()

        self.data_instance = data_instance
