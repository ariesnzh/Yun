import pandas as pd
from typing import List


def voting_ensemble(pred_res_list: List[pd.DataFrame], weights: List[float] = None, is_rank: bool = False, is_avg: bool = False):
	if is_rank:
		pred_res_list = [df.rank(pct=True, axis=1) for df in pred_res_list]
	if is_avg:
		weights = [1.0 / len(pred_res_list) for _ in range(len(pred_res_list))]
	results = sum(df * weight for df, weight in zip(pred_res_list, weights))
	return results


if __name__ == "__main__":
	dfs = [pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]}), 
		pd.DataFrame({'A': [7, 8, 9], 'B': [10, 11, 12]}), 
		pd.DataFrame({'A': [13, 14, 15], 'B': [16, 17, 18]})]
	weights = [0.5, 0.3, 0.2]
	res = voting_ensemble(dfs, weights, is_rank=True)
	res = voting_ensemble(dfs, is_rank=False, is_avg=True)
	