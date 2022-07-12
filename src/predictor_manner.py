import configures_manner
import json
from data_preparation import DataPreparation
import pandas as pd
import datetime
from models import miae_type1, miae_type2, miae_type3


class Predictor:
    def __init__(self, model_predictor_ae_type: str):
        super().__init__()
        self.model_predictor_ae_type = model_predictor_ae_type

    def generate_type1_instance(self):
        model_instance = miae_type1.MIAET1(self.model_hyperparameters)
        return model_instance

    def generate_type2_instance(self):
        model_instance = miae_type2.MIAET2(self.model_hyperparameters)
        return model_instance

    def generate_type3_instance(self):
        model_instance = miae_type3.MIAET3(self.model_hyperparameters)
        return model_instance

    def load_instace_model_from_id(self, model_id: str):
        with open(configures_manner.metadata_path + model_id + ".json") as json_file:
            self.model_metadata = json.load(json_file)
        configures_manner.add_all_configures_to_globals(self.model_metadata)
        self.model_hyperparameters = self.model_metadata["params"]
        self.model_instance = getattr(
            self, f"generate_{self.model_predictor_ae_type}_instance"
        )()
        self.model_instance.load_instace_model_from_id(model_id)

    def gen_data_to_predict(self, begin: str, end: str):
        configures_manner.add_variable_to_globals("begin", begin)
        configures_manner.add_variable_to_globals("end", end)
        data_obj = DataPreparation()
        data_obj.get_data_to_web_request(
            configures_manner.repo,
            configures_manner.path,
            configures_manner.inputFeatures,
            configures_manner.inputWindowSize,
            configures_manner.begin,
            configures_manner.end,
        )
        data_obj.outseqlen = self.model_hyperparameters["outseqlen"]
        data_obj.inseqlen = self.model_hyperparameters["inseqlen"]
        data_obj.data_to_test_create()
        return data_obj.data_to_test

    def predict(self, data_to_predict):
        yhat = self.model_instance.predicting(data_to_predict)
        return yhat.reshape(-1).numpy()

    def predictions_to_weboutput(self, yhat, begin, end):
        period = pd.date_range(begin, end)
        returned_dictionary = list()
        for date, value in zip(period, yhat):
            returned_dictionary.append(
                {
                    "date": datetime.datetime.strftime(date, "%Y-%m-%d"),
                    "prediction": str(value),
                }
            )
        return returned_dictionary
