import sys
from urllib import response

sys.path.append("../src")

import configures_manner
import evaluator_manner

repo = "p971074907"
path = "brl:rn"
modelType = "type2"

metadata_to_train = {
    "schema": "schema",
    "entity": "autoencoder",
    "inputFeatures": "date:newDeaths",
    "outputFeatures": "newDeaths",
    "begin": "2020-06-01",
    "end": "2020-7-09",
    "inputWindowSize": 7,
    "outputWindowSize": 7,
    "testSize": 14,
    "modelConfigs": {
        "inseqlen": 7,
        "outseqlen": 7,
        "growth": 4,
        "latent_space_dim": 7,
        "n_features": 1,
        "n_targets": 1,
        "activation": "ReLU",
        "epochs": 10,
        "seed": [51, 41, 15],
        "learning_rate": 0.0005,
    },
}

configures_manner.add_variable_to_globals("model_type", modelType)
configures_manner.add_variable_to_globals("repo", repo)
configures_manner.add_variable_to_globals("path", path)
configures_manner.overwrite(metadata_to_train)

evaluator = evaluator_manner.Evaluator()
evaluator.train()
