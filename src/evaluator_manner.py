from data_preparation import DataPreparation
from models.miae_type1 import MIAET1, RMSELoss
from models.miae_type2 import MIAET2
from models.miae_type3 import MIAET3
import configures_manner
import itertools
import json
from datetime import date, datetime


class Evaluator:
    def __init__(self):
        super().__init__()
        self._set_configures_combinations()
        self.gen_instaces_to_train()

    def generate_type1_instance(self, combination):
        model_instance = MIAET1(combination)
        return model_instance

    def generate_type2_instance(self, combination):
        model_instance = MIAET2(combination)
        return model_instance

    def generate_type3_instance(self, combination):
        model_instance = MIAET3(combination)
        return model_instance

    def gen_instaces_to_train(self):
        self.model_list = [
            getattr(self, f"generate_{configures_manner.model_type}_instance")(
                combination
            )
            for combination in self.configures_combinations
        ]

    def train(self):
        models_metadatas_to_save = {}
        for model in self.model_list:
            self.model = model
            self.gen_data_to_train()
            model.train(self.data_instance)
            model_metadata_dict = model._gen_model_metada()
            models_metadatas_to_save[
                model_metadata_dict["instance_id"]
            ] = model_metadata_dict
        # print(models_metadatas_to_save)
        self.save_search_json(models_metadatas_to_save)

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

    def _set_configures_combinations(self) -> list():
        configures_variation = list()
        for value in configures_manner.modelConfigs.values():
            if isinstance(value, list):
                configures_variation.append(value)
            else:
                configures_variation.append([value])

        configures_combinations = list(itertools.product(*configures_variation))

        self.configures_combinations = [
            self.gen_instance_dict(combination)
            for combination in configures_combinations
        ]

        return configures_combinations

    def gen_instance_dict(self, combination) -> dict():
        combination_dict = {
            configure_name: combination[index]
            for index, configure_name in enumerate(configures_manner.modelConfigs)
        }
        return combination_dict

    def save_search_json(self, search_dictionary_to_save):
        with open(
            configures_manner.metadata_path
            + "search-"
            + datetime.now().strftime("%Y-%m-%d-" "%H:%M:%S")
            + ".json",
            "w",
        ) as fp:
            json.dump(search_dictionary_to_save, fp, indent=4)
        return search_dictionary_to_save
