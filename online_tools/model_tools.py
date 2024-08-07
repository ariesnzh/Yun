import importlib
import os
from collections import OrderedDict
from loguru import logger
from omegaconf import DictConfig
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common.global_parameters import Model_Module_Map


class DLInferenceInterface:
	def __init__(self, model_type: str, model_path: str, model_hp: DictConfig, device: str = 'gpu') -> None:
		self.model_type = model_type
		self.model_path = model_path
		self.model_hp = model_hp
		self.device = torch.device("cuda" if torch.cuda.is_available() and device == 'gpu' else 'cpu')
		self.model = self.init_model()
		self.load_model()
	
	def init_model(self) -> nn.Module:
		module = importlib.import_module("models")
		model = getattr(module, Model_Module_Map[self.model_type])(**self.model_hp)
		return model
	
	def load_model(self):
		checkpoint = torch.load(self.model_path)
		params = OrderedDict([('.'.join(k.split('.')[1:]), v) for (k, v) in checkpoint["state_dict"].items()])
		self.model.load_state_dict(params)
		self.model.to(self.device)
	
	def inference(self, dataloader: DataLoader):
		self.model.eval()
		inference_outputs = []
		with torch.inference_mode(mode=True):
			inference_thread = tqdm(dataloader, desc="Inference", leave=True)
			for _, batch in enumerate(inference_thread):
				inputs = batch.to(device=self.device, dtype=torch.float)
				y_preds = self.model(inputs)
				inference_outputs.extend(y_preds.cpu().detach().numpy().ravel())
		return inference_outputs


def get_latest_file(directory):
	all_files = [os.path.join(directory, f) for f in os.listdir(directory)]
	latest_file = max(all_files, key=os.path.getctime)
	return latest_file
