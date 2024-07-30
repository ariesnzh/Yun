import os
import re
import csv
import argparse
import pandas as pd
from typing import List
from datadeal import read_factor_data


def load_feature_names(file: str) -> List[str]:
	try:
		with open(file, "r") as f:
			reader = csv.reader(f)
			feature_names = [row[0].strip() for row in reader]
		return feature_names
	except FileNotFoundError:
		raise FileNotFoundError(f"Error: The file '{file}' was not found.")
	except Exception as e:
		raise RuntimeError(f"An error occurred: {e}")


def get_factors_from_dataset(save_dir, feature_names, authors, start, end):
	if not os.path.exists(save_dir):
		os.makedirs(save_dir, exist_ok=True)
	illegal_factors = []
	for factor, name in zip(feature_names, authors):
		print(factor)
		df: pd.DataFrame = read_factor_data(factor_name=factor, author=name, start=start, end=end)
		if df.empty:
			illegal_factors.append(factor)
		else:
			df.index = df.index.strftime('%Y-%m-%d')
			df.to_pickle(f'{save_dir}/{factor}.pkl')
	print(illegal_factors)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--feature_file', default='/data/nzh/model/config/all_pool_factor_list.csv', type=str)
	parser.add_argument('--save_dir', default='/data/nzh/model/raw_data/daily/strategy5', type=str)
	parser.add_argument('--start', default='2018-01-01', type=str)
	parser.add_argument('--end', default='2024-06-28', type=str)
	args = parser.parse_args()

	feature_names = load_feature_names(file=args.feature_file)
	authors = [re.compile('[a-zA-Z]*').match(feature.split('_')[1]).group() for feature in feature_names]
	get_factors_from_dataset(save_dir=args.save_dir, feature_names=feature_names, authors=authors, start=args.start, end=args.end)
