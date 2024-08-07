import os
import re
import hydra
import joblib
import datetime
import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import datetime
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from typing import List

from pytorch_tabnet.tab_model import TabNetRegressor
from torch.utils.data import DataLoader

import dstool
from online_tools.ensemble_tools import voting_ensemble
from online_tools.data_tools import *
from common.global_parameters import *
from online_tools.model_tools import *
from online_tools.time_tools import get_k_trading_days_before, get_next_trading_day


ensemble_result = None
output_path = None
predictions_list = []

inference_date: str = None  # 开启推理日期（需要预测日期之前）
cache_dir: str = None


class Deployment:
	def __init__(self, factor_list: List[str], inference_date: str, model_id: str, model_path: str, model_hp: DictConfig, seq_len=1, bias: int = 0):
		self.factor_list = factor_list
		self.inference_date = inference_date
		self.model_id = model_id
		self.model_path = model_path
		self.model_hp = model_hp
		# 预测13号，当前是12号，seq_len=5，则取12/11/10/9/8五天因子值用于预测
		self.seq_len = seq_len
		" 近期数据有问题，为测试临时调整区间，后续删除 "
		day_list = get_k_trading_days_before(dt=inference_date, k=self.seq_len + bias)
		self.start_date = day_list[0].strftime('%Y-%m-%d')
		self.end_date = day_list[self.seq_len - 1].strftime('%Y-%m-%d')
		# day_list = get_k_trading_days_before(dt=inference_date, k=self.seq_len)
		# self.start_date = day_list[0].strftime('%Y-%m-%d')
		# self.end_date = day_list[-1].strftime('%Y-%m-%d')
		# print(self.start_date, self.end_date)
		# self.end_date = day_list[self.seq_len - 1].strftime('%Y-%m-%d')
		# print(self.end_date)
		self.next_date = get_next_trading_day(dt=inference_date).strftime('%Y-%m-%d')
		logger.info(f'History data start date: {self.start_date}')
		logger.info(f'History data end date: {self.end_date}')
		logger.info(f'Next date: {self.next_date}')
		self.preprocess_data = self.get_preprocess_data()
	
	def get_preprocess_data(self):
		logger.info('Preprocess data')
		factor_preprocess = pd.DataFrame()
		for name in tqdm(self.factor_list):
			author = re.search(r'.*?_([a-zA-Z]*)[0-9]+',name).group(1)
			factor = dstool.datadeal.read_factor_data(factor_name=name, author=author, start=self.start_date, end=self.end_date)
			factor.index = pd.to_datetime(factor.index).strftime('%Y-%m-%d')
			if factor.index[-1] != self.end_date:
				raise ValueError(f'The {name} values have not been updated today!')
			factor_preprocess = pd.concat([factor_preprocess, preprocessing_thread(factor, name)], axis=1)
		factor_preprocess = factor_preprocess.T.sort_index().T
		factor_preprocess = factor_preprocess.reset_index().rename({'level_0': DATE_NAME, 'level_1': SECURITY_NAME}, axis=1).sort_values([DATE_NAME, SECURITY_NAME])
		if factor_preprocess.isna().sum().sum() != 0:
			raise ValueError('factor_preprocess_df has NaN !')
		""" 保存到home目录下 """
		# cache_save_path = Path(os.path.join(os.path.dirname(__file__), 'cache_daily', self.inference_date))
		# if not cache_save_path.exists():
		# 	cache_save_path.mkdir(parents=True, exist_ok=True)
		# factor_preprocess.to_pickle(os.path.join(cache_save_path, 'factor_preprocess.pkl'))
		""" 保存到data目录下 """
		cache_save_path = Path(os.path.join(DATA_SAVE_PATH, 'cache_daily', self.inference_date))
		if not cache_save_path.exists():
			cache_save_path.mkdir(parents=True, exist_ok=True)
		factor_preprocess.to_pickle(os.path.join(cache_save_path, 'factor_preprocess.pkl'))
		""" 保存到schedule下 """
		cache_save_path = Path(os.path.join(SCHEDULE_RES_SAVE_PATH, 'cache_daily', self.inference_date))
		if not cache_save_path.exists():
			cache_save_path.mkdir(parents=True, exist_ok=True)
		factor_preprocess.to_pickle(os.path.join(cache_save_path, 'factor_preprocess.pkl'))
		return factor_preprocess
	
	def save_predictions(self, preds: pd.DataFrame):
		""" 保存到home目录下 """
		# cache_save_path = Path(os.path.join(os.path.dirname(__file__), 'cache_daily', self.inference_date))
		# if not cache_save_path.exists():
		# 	cache_save_path.mkdir(parents=True, exist_ok=True)
		# output_file = f'pred_{self.model_id}_result.pkl'
		# preds.to_pickle(os.path.join(cache_save_path, output_file))

		""" 保存到data目录下 """
		cache_save_path = Path(os.path.join(RES_SAVE_PATH, 'cache_daily', self.inference_date))
		if not cache_save_path.exists():
			cache_save_path.mkdir(parents=True, exist_ok=True)
		output_file = f'pred_{self.model_id}_result.pkl'
		preds.to_pickle(os.path.join(cache_save_path, output_file))
		""" 保存到schedule用户下 """
		global cache_dir
		if cache_dir:
			cache_save_path = Path(cache_dir)
		else:
			cache_save_path = Path(os.path.join(SCHEDULE_RES_SAVE_PATH, 'cache_daily', self.inference_date))
		if not cache_save_path.exists():
			cache_save_path.mkdir(parents=True, exist_ok=True)
		output_file = f'pred_{self.model_id}_result.pkl'
		preds.to_pickle(os.path.join(cache_save_path, output_file))
	
	def lightgbm_inference(self):
		model = joblib.load(self.model_path)
		features, index = NoneBatchDataset(data=self.preprocess_data, date_column=DATE_NAME, security_column=SECURITY_NAME, feature_columns=self.factor_list).construct()
		preds = model.predict(features.values, model.best_iteration_)
		stacked_array = np.stack([np.array([item[0] for item in index.values]),
								  np.array([item[1] for item in index.values]),
								  preds], axis=1)
		res = pd.DataFrame(stacked_array, columns=[DATE_NAME, SECURITY_NAME, f'pred_label'])
		res = -res.pivot(index=DATE_NAME, columns=SECURITY_NAME, values=f'pred_label').astype(np.float64)
		return res
	
	def xgboost_inference(self):
		# 模型加载
		model = xgb.XGBRegressor(**self.model_hp)
		booster = xgb.Booster()
		booster.load_model(self.model_path)
		model._Booster = booster
		# model = xgb.Booster(model_file=self.model_path)
		# 数据集划分（标签和特征）
		features, index = NoneBatchDataset(data=self.preprocess_data, date_column=DATE_NAME, security_column=SECURITY_NAME, feature_columns=self.factor_list).construct()
		# 预测
		# features = xgb.DMatrix(features)
		preds = model.predict(features.values)
		# 结果保存
		stacked_array = np.stack([np.array([item[0] for item in index.values]),
								  np.array([item[1] for item in index.values]),
								  preds], axis=1)
		res = pd.DataFrame(stacked_array, columns=['date', 'code', f'pred_label'])
		res = -res.pivot(index='date', columns='code', values=f'pred_label').astype(np.float64)
		return res

	def tabnet_inference(self):
		# 模型加载
		model = TabNetRegressor(**self.model_hp)
		model.load_model(filepath=self.model_path)
		# 数据集划分（标签和特征）
		features, index = NoneBatchDataset(data=self.preprocess_data, date_column=DATE_NAME, security_column=SECURITY_NAME, feature_columns=self.factor_list).construct()
		# 预测
		preds = model.predict(features.values)
		# 结果保存
		stacked_array = np.stack([np.array([item[0] for item in index.values]),
								  np.array([item[1] for item in index.values]),
								  preds.squeeze()], axis=1)
		res = pd.DataFrame(stacked_array, columns=['date', 'code', f'pred_label'])
		res = -res.pivot(index='date', columns='code', values=f'pred_label').astype(np.float64)
		return res

	def dl_inference(self):
		inference_X, inference_index = generate_samples(feature_label_df=self.preprocess_data, seq_len=self.seq_len, feature_name=self.factor_list, date_name=DATE_NAME, security_name=SECURITY_NAME)
		inference_dataset = PredTimeSeriesBatchDataset(X=inference_X)
		inference_dataloader = DataLoader(dataset=inference_dataset, batch_size=1024, shuffle=False, num_workers=2)
		inference_engine = DLInferenceInterface(model_type=self.model_id.split('_')[0], model_path=self.model_path, model_hp=self.model_hp, device='gpu')
		predictions = inference_engine.inference(dataloader=inference_dataloader)
		res = pd.DataFrame(np.array([
			[item[0] for item in inference_index],
			[item[1] for item in inference_index],
			predictions]).T, columns=[DATE_NAME, SECURITY_NAME, f"pred_label"])
		res = -res.pivot(index='date', columns='code', values=f"pred_label").astype(np.float64)
		return res
	
	def inference(self):
		logger.info(f'Start inference {self.model_id}')
		if 'lightgbm' in self.model_id:
			res = self.lightgbm_inference()
		elif 'xgboost' in self.model_id:
			res = self.xgboost_inference()
		elif 'tabnet' in self.model_id:
			res = self.tabnet_inference()
		elif 'tcn' in self.model_id:
			res = self.dl_inference()
		else:
			raise KeyError(f'Invalid model id {self.model_id}')
		self.save_predictions(res)
		return res


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def predict(cfg: DictConfig):
	global inference_date
	if inference_date:
		today = inference_date
	else:
		today = datetime.today().strftime('%Y-%m-%d')
	factor_list_file = get_latest_file(directory=FACTOR_LIST_PATH)
	logger.info(f'Factor list file:{factor_list_file}')
	factor_list = pd.read_csv(factor_list_file, header=None).iloc[:, 0]
	# factor_list = pd.read_csv(FACTOR_LIST_PATH, header=None).iloc[:, 0]
	# 遍历每一个基模型
	# predictions_list = []
	for model_id in cfg.models:
		logger.info(f'Model ID: {model_id}')
		# 获取该基模型超参数
		hp = cfg.models[model_id]
		if hp['model_path'] is None:
			model_path = get_latest_file(directory=hp['model_dir'])
		else:
			model_path = hp['model_path']
		logger.info(f'Model path: {model_path}')
		dp = Deployment(factor_list=factor_list, inference_date=today, model_id=model_id, model_path=model_path, model_hp=hp['hyper_parameter'], seq_len=hp['seq_len'], bias=0)
		res = dp.inference()
		predictions_list.append(res)
		logger.info(f'Added results of {model_id}')
	global ensemble_result
	ensemble_result = voting_ensemble(predictions_list, weights=cfg['ensemble_weights'], is_rank=True)
	""" 保存到home目录下 """
	# cache_save_path = Path(os.path.join(os.path.dirname(__file__), 'cache_daily', today))
	# if not cache_save_path.exists():
	# 	cache_save_path.mkdir(parents=True, exist_ok=True)
	# output_path = os.path.join(cache_save_path, f'ensemble_result.pkl')
	# ensemble_result.to_pickle(output_path)
	# logger.info(f'Save ensemble result: {output_path}')
	""" 保存到data目录下 """
	global output_path
	cache_save_path = Path(os.path.join(RES_SAVE_PATH, 'cache_daily', today))
	if not cache_save_path.exists():
		cache_save_path.mkdir(parents=True, exist_ok=True)
	output_path = os.path.join(cache_save_path, f'ensemble_result.pkl')
	ensemble_result.to_pickle(output_path)
	logger.info(f'Save ensemble result: {output_path}')
	""" 保存到schedule用户下 """
	global cache_dir
	if cache_dir:
		cache_save_path = Path(cache_dir)
	else:
		cache_save_path = Path(os.path.join(SCHEDULE_RES_SAVE_PATH, 'cache_daily', today))
	if not cache_save_path.exists():
		cache_save_path.mkdir(parents=True, exist_ok=True)
	output_path = os.path.join(cache_save_path, f'ensemble_result.pkl')
	ensemble_result.to_pickle(output_path)
	logger.info(f'Save ensemble result: {output_path}')
	# print(ensemble_result)
	# print(output_path)
	# print(predictions_list)
	return ensemble_result, output_path, predictions_list


def online_interface(date: str = None, output_dir: str = None):
	"""
	date: 开启推理的日期(不同于需要预测的日期，在需要预测的日期之前)
	output_dir: 数据、结果缓存文件保存路径(使用时在cache_dir下默认会创建以日期命名的子文件夹,在该子文件夹中保存文件)
	"""
	if date:
		global inference_date
		inference_date = date
	if output_dir:
		global cache_dir
		cache_dir = output_dir
	predict()
	return ensemble_result, output_path, predictions_list


if __name__ == '__main__':
	logger.add(f"{LOG_BASE_PATH}/{datetime.now().strftime('%Y-%m-%d_%H_%M_%S')}.log")
	res = online_interface()
	print(res)