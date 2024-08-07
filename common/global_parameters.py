""" Global hyper parameters """
DATE_NAME = 'date'

SECURITY_NAME = 'code'

CONFIG_NAME = 'config'

# FACTOR_LIST_PATH = '/data/nzh/ylmodel_online/config/all_pool_factor_list.csv'
RES_SAVE_PATH = '/data/nzh/ylmodel_online/res'
SCHEDULE_RES_SAVE_PATH = '/home/schedule/model_position_deploy/ptl_stock'
DATA_SAVE_PATH = '/data/nzh/ylmodel_online/data'
LOG_BASE_PATH = '/data/nzh/ylmodel_online/logs/loguru'
CONFIG_PATH = '/data/nzh/ylmodel_online/config'
FACTOR_LIST_PATH = '/data/nzh/ylmodel_online/factor'

N_JOBS = 32

Model_Module_Map = {
	'tcn': 'TCNModel'
}
