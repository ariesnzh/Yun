import sys 
sys.path.append("/data/public/arrangement_buffer_v1/arrangement_buffer/factor_gen")
import os
from dstool.CONSTANT import HDF_FATCOR_FLODER
import dstool.QuantStudio.api as QS
# from QuantStudio.api import QS
import dstool.tools
# from push2hdf5.CONSTANT import HDF_FATCOR_FLODER
# import push2hdf5.QuantStudio.api as QS
import glob
# from push2hdf5 import tools
import pandas as pd
import re

from pandas import DataFrame

def wirte_factor_data(df: DataFrame, factor_name:str, author: str, ):
    '''
    存新因子和更新已有因子都可以
    '''
    df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated()]
    factor_floder_connect = QS.FactorDB.HDF5DB(sys_args={"主目录":HDF_FATCOR_FLODER}).connect()
    # author_connect = factor_floder_connect[author]
    factor_floder_connect.writeFactorData(df, author, factor_name)

def read_factor_data(factor_name:str, author: str, start: str, end: str):
    '''
    读取因子的数据
    '''

    start, end = dstool.tools.deal_start_end_time(start, end)
    factor_floder_connect = QS.FactorDB.HDF5DB(sys_args={"主目录":HDF_FATCOR_FLODER}).connect()
    author_factor_connect = factor_floder_connect[author][factor_name]

    dt_index = author_factor_connect.getDateTime(start_dt=start, end_dt=end)
    
    return author_factor_connect[dt_index,:]

def wirte_one_day_factor_data(df: DataFrame, date: str):
    '''

    储存 每天的因子 数据 
    date  YYYY-MM-DD/YYYYMMDD/YYYYMMDDHHMMSS 的形式的字符串
    '''
    factor_floder_connect = QS.FactorDB.HDF5DB(sys_args={"主目录":HDF_FATCOR_FLODER}).connect()
    
    author_pattern = re.compile('.*?_([a-zA-Z]*)[0-9]+')

    date = dstool.tools.deal_str_datetime(date)

    df_name_list = df.index.to_list()
    author_list = [author_pattern.match(i) for i in df_name_list]
    author_list = [i.groups()[0] if i else 'wrong' for i in author_list]
    for i in range(len(df_name_list)):
        factor_name = df_name_list[i]
        author = author_list[i]
        if author == 'wrong':
            raise Exception('factor_name is wrong')
        tmpdf = df.loc[[factor_name], :]
        tmpdf.index = [date]
    
        factor_floder_connect.writeFactorData(tmpdf, author, factor_name)

if __name__ =="__main__":
# 存入因子
    # # 指定要遍历的文件夹路径
    # folder_path = '/data/public/arrangement_buffer_v1/arrangement_buffer/factor_gen/temp_save/zwj'
    # from tqdm import tqdm

    # # 使用glob模块匹配Alpha开头所有的csv文件
    # csv_files = glob.glob(folder_path + "/Alpha*.csv")
    # for alpha_value_path in tqdm(csv_files):
    #     alpha_name = os.path.split(alpha_value_path)[-1].strip('.csv')
    #     alpha_df = pd.read_csv(alpha_value_path,index_col = 0,engine='pyarrow')
    #     match = re.search(r'.*?_([a-zA-Z]*)[0-9]+', alpha_name)
    #     author = match.group(1)
    #     print(alpha_name,author)
    #     wirte_factor_data(df = alpha_df,factor_name = alpha_name,author = author)


# # 更新一天
#     one_day_alpha_path = '/data/public/arrangement_buffer_v1/arrangement_buffer/factor_gen/temp_save/2022-01-04.csv'
#     one_df_alpha = pd.read_csv(one_day_alpha_path,index_col = 0)
#     dt = os.path.split(one_day_alpha_path)[-1].strip('.csv')
#     wirte_one_day_factor_data(one_df_alpha ,dt )
#     print(dt)


# # 查看
#      # 指定要遍历的文件夹路径
#     folder_path = ''
#     # 使用glob模块匹配Alpha开头所有的csv文件
#     csv_files = glob.glob(folder_path + "/Alpha*.csv")
#     for alpha_value_path in csv_files:
#         alpha_name = os.path.split(alpha_value_path)[-1].strip('.csv')
#         alpha_df = pd.read_csv(alpha_value_path,index_col = 0)
#         match = re.search(r'.*?_([a-zA-Z]*)[0-9]+', alpha_name)
#         author = match.group(1)
#         print(alpha_name)
#         df = read_factor_data(factor_name = alpha_name,author = author,start="2020-01-01",end = "2024-02-15")
#         print(df)

  
    # alpha_df = pd.read_csv("/data/public/arrangement_buffer_v1/arrangement_buffer/factor_gen/temp_save/zwj1/Alpha_zjx032.csv",index_col = 0,engine='pyarrow')

    # wirte_factor_data(df = alpha_df,factor_name = "Alpha_zjx032",author = "zjx")

    df = read_factor_data(factor_name = "Alpha_zjx035",author = 'zjx',start="2022-05-10",end = "2022-05-23")
    print(df.dropna(how='all').tail(40))

