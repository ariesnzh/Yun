import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from loguru import logger
from tqdm import tqdm
from typing import List, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset

from common.global_parameters import N_JOBS


def data_normlize(panel):
	panel = panel.apply(lambda x: (x-x.mean())/x.std(), axis=1)
	return panel

def missing_fill(panel):
	panel = panel.ffill()
	panel = panel.T.fillna(panel.T.mean()).T
	return panel

def Panel2Series(panel, col_name):
	stack_df = panel.stack().to_frame()
	stack_df.columns = [col_name]
	return stack_df

def preprocessing_thread(pannel, factor_name):
	pannel = pannel.replace([np.inf, -np.inf], np.nan)
	pannel = data_normlize(pannel)
	pannel = missing_fill(pannel)
	series = Panel2Series(pannel, factor_name)
	return series


class NoneBatchDataset:
	def __init__(self,
				 data: pd.DataFrame,
				 date_column: str,
				 security_column: str,
				 feature_columns: List[str] = None) -> None:
		self.data: pd.DataFrame = data
		self.date_column: str = date_column
		self.security_column: str = security_column
		if feature_columns is not None:
			self.feature_columns = feature_columns
		else:
			self.feature_columns = data.columns.drop([date_column, security_column]).to_list()
	
	def construct(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
		data = self.data[self.feature_columns]
		time_stock_tags = self.data[[self.date_column, self.security_column]]
		return data, time_stock_tags


class PredTimeSeriesBatchDataset(Dataset):
	def __init__(self, X: np.ndarray) -> None:
		self.X = X
	
	def __len__(self) -> int:
		return self.X.__len__()
	
	def __getitem__(self, index) -> Tensor:
		return torch.from_numpy(self.X[index]).float()


def generate_samples(feature_label_df: pd.DataFrame, seq_len: int, feature_name: List[str], date_name: str, security_name: str):
	results = Parallel(n_jobs=N_JOBS)(delayed(generate_samples_thread_pred)(code, group_code, seq_len, feature_name, date_name, security_name) 
								for code, group_code in tqdm(feature_label_df.groupby(security_name), desc='generating samples'))
	data_x, data_index = [], []
	for res in results:
		data_x.extend(res[0])
		data_index.extend(res[1])
	return np.array(data_x), np.array(data_index)


def generate_samples_thread_pred(code, group_code, seq_len: int, feature_name: List[str], date_name: str, security_name: str):
	data_x, data_index = [],  []
	group_code = group_code.sort_values(date_name)
	for i in range(len(group_code) - seq_len + 1):
		data_x.append(group_code.iloc[i: i + seq_len][feature_name].values)
		data_index.append(group_code.iloc[i + seq_len - 1][[date_name, security_name]].values)
	return data_x, data_index
