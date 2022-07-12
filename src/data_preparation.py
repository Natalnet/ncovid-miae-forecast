import torch
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import datetime


class DataPreparation:
    def __init__(self):
        super().__init__()

    def get_data(self, repo, path, feature, window_size, begin, end):
        if begin == None:
            begin = "2020-01-01"
        if end == None:
            end = "2050-01-01"

        self.window_size = int(window_size)

        file_name = "".join(
            f"http://ncovid.natalnet.br/datamanager/"
            f"repo/{repo}/"
            f"path/{path}/"
            f"features/{feature}/"
            f"window-size/{window_size}/"
            f"begin/{begin}/"
            f"end/{end}/as-csv"
        )
        df = pd.read_csv(file_name, parse_dates=["date"], index_col="date")

        self.input_features = [x for x in feature.split(":")[1:]]
        self.targets_features = self.input_features[0]

        data = df
        # store first and last day available
        self.begin_raw = df.index[0]
        self.end_raw = df.index[-1]
        self.data = data
        self.instance_region = path
        return data

    def get_data_to_web_request(self, repo, path, feature, window_size, begin, end):
        if begin == None:
            begin = "2020-01-01"
        if end == None:
            end = "2050-01-01"

        self.window_size = int(window_size)

        begin_delay = datetime.datetime.strptime(
            begin, "%Y-%m-%d"
        ) - datetime.timedelta(days=self.window_size)
        new_begin = datetime.datetime.strftime(begin_delay, "%Y-%m-%d")

        period_date_request = pd.date_range(new_begin, end)
        date_gap_to_assert_length = len(period_date_request) % self.window_size
        new_end_datetime = datetime.datetime.strptime(
            end, "%Y-%m-%d"
        ) + datetime.timedelta(days=date_gap_to_assert_length)

        new_end = datetime.datetime.strftime(new_end_datetime, "%Y-%m-%d")

        file_name = "".join(
            f"http://ncovid.natalnet.br/datamanager/"
            f"repo/{repo}/"
            f"path/{path}/"
            f"features/{feature}/"
            f"window-size/{window_size}/"
            f"begin/{new_begin}/"
            f"end/{new_end}/as-csv"
        )

        df = pd.read_csv(file_name, parse_dates=["date"], index_col="date")

        self.input_features = [x for x in feature.split(":")[1:]]
        self.targets_features = self.input_features[0]

        data = df
        # store first and last day available
        self.begin_raw = df.index[0]
        self.end_raw = df.index[-1]
        self.data = data
        self.instance_region = path
        return data

    def data_tensor_generate(self, output_len):

        self.inseqlen = self.window_size
        if output_len:
            self.outseqlen = output_len

        window_dataset = self.data.iloc[: -self.outseqlen].rolling(
            self.inseqlen, min_periods=1, win_type=None, center=False
        )

        inputs, targets = [], []
        for window in window_dataset:
            if len(window) == self.inseqlen:
                inpt = window[self.input_features]
                inputs.append(inpt.T.values)
                last_inpt_day = pd.Timestamp(inpt.index[-1])
                next_day_after_last = last_inpt_day + pd.DateOffset(1)
                start_pred_range = next_day_after_last
                data_forward_range = pd.date_range(
                    start=start_pred_range, periods=self.outseqlen
                )
                trg = self.data[self.targets_features].loc[data_forward_range]
                targets.append([trg.T.values])

        self.inputs_tensor = torch.Tensor(np.array(inputs))
        self.targets_tensor = torch.Tensor(np.array(targets))

    def set_as_train_data(self, x, y):
        self.X_train = x
        self.Y_train = y

    def set_as_test_data(self, x, y):
        self.X_test = x
        self.Y_test = y

    def train_test_split_by_percent(self, prct_to_train=0.8):
        self.percent_to_train = prct_to_train
        x_to_train = self.inputs_tensor[: int(len(self.inputs_tensor) * prct_to_train)]
        y_to_train = self.targets_tensor[
            : int(len(self.targets_tensor) * prct_to_train)
        ]
        self.set_as_train_data(x_to_train, y_to_train)

        x_to_test = self.inputs_tensor[int(len(self.inputs_tensor) * prct_to_train) :]
        y_to_test = self.targets_tensor[int(len(self.targets_tensor) * prct_to_train) :]
        self.set_as_test_data(x_to_test, y_to_test)

    def train_test_split_by_days(self, test_size):
        self.test_size = int(test_size / self.inseqlen)
        x_to_train = self.inputs_tensor[: -self.test_size]
        y_to_train = self.targets_tensor[: -self.test_size]
        self.set_as_train_data(x_to_train, y_to_train)
        x_to_test = self.inputs_tensor[-self.test_size :]
        y_to_test = self.targets_tensor[-self.test_size :]
        self.set_as_test_data(x_to_test, y_to_test)

    def dataloader_train_generate(self):
        self.data_train_tensor = TensorDataset(self.X_train, self.Y_train)
        self.data_train = DataLoader(
            self.data_train_tensor, batch_size=self.batch_size, shuffle=False
        )

    def dataloader_test_generate(self):

        self.data_test_tensor = TensorDataset(self.X_test, self.Y_test)
        self.data_test = DataLoader(
            self.data_test_tensor, batch_size=self.batch_size, shuffle=False
        )

    def dataloader_create(self, batch_size=None):
        self.batch_size = batch_size
        if self.batch_size == None:
            self.batch_size = int(len(self.X_train) / 3)

        self.dataloader_train_generate()
        self.dataloader_test_generate()

    def data_to_test_create(self, data_test=None):
        if data_test is not None:
            self.data_to_test = data_test
        else:
            self.data_to_test = self.data

        data_test_vect = []
        for i in range(0, len(self.data_to_test), self.outseqlen):
            window = self.data_to_test[self.input_features][i : i + self.inseqlen]
            if len(window) == self.inseqlen:
                data_test_vect.append(window.T.values)

        self.data_to_test = torch.Tensor(np.array(data_test_vect))

    def data_split_by_feature(self):
        features_splitted = []
        for i in range(self.X_train.shape[1]):
            ft = torch.Tensor(np.array([input[i].numpy() for input in self.X_train]))
            ft_dataset = TensorDataset(ft, ft)
            ft_dataloader = DataLoader(
                ft_dataset, batch_size=self.batch_size, shuffle=False
            )
            features_splitted.append(ft_dataloader)

        self.splited_data = features_splitted

    def att_data(self, new_features, new_targets_features, data=None):
        if data:
            self.data = data
        self.features = new_features
        self.targets_features = new_targets_features

        self.data_tensor_generate(self.inseqlen, self.outseqlen)
        self.train_test_split(self.percent_to_train)
        self.dataloader_create(self.batch_size)
        self.data_split_by_feature()
