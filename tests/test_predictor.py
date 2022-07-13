import sys

sys.path.append("../src")

from predictor_manner import Predictor

begin = "2020-03-13"
end = "2020-07-15"

modelCategory = "type3"
modelInstance = "a62cd526-ff16-11ec-912c-5d81233e93b4"

predictor = Predictor("type1")
predictor.load_instace_model_from_id(modelInstance)


data_to_predict = predictor.gen_data_to_predict(begin, end)

yhat = predictor.predict(data_to_predict)

yhat_json = predictor.predictions_to_weboutput(yhat, begin, end)

print(yhat_json)
