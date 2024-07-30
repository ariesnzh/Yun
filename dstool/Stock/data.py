import dstool.QuantStudio.api as QS
from dstool.CONSTANT import HDF_STOCK_FLODER
from dstool.tools import deal_start_end_time

from pandas import DataFrame
import os

__all__ = ['get_factor_expose_data', 'Quote_period_data']

def get_factor_expose_data(floder: str, field:str, code: list[str]=['*'], start:str='', end:str=''):
    '''
    floder: factor_descriptor_exposure  or factor_exposure
    
    field 查看 /data/all_data/QuoteData/FACTOR/FinFactor 下的文件名称
    '''
    if not isinstance(code, list): raise Exception('code type is wrong')
    start, end = deal_start_end_time(start, end)
    
    period_connect = QS.FactorDB.HDF5DB(sys_args={"主目录":'/data/all_data/QuoteData/FACTOR/FinFactor'}).connect()[floder]
    
    dt_index = period_connect[field].getDateTime(start_dt=start, end_dt=end)
    if '*' in code:
        return period_connect[field][dt_index, :]
    else:
        return period_connect[field][dt_index, code]

class Quote_period_data():
    def __init__(self, period: str) -> None:
        
        if period not in os.listdir(HDF_STOCK_FLODER):
            raise Exception('period is wrong')
        
        self.period = period
        
        self.period_connect = QS.FactorDB.HDF5DB(sys_args={"主目录":HDF_STOCK_FLODER}).connect()[self.period]
    
    @property
    def field(self) -> list:
        return self.period_connect.FactorNames
        
    def get_data(self, field:str, code: list[str]=['*'], start:str='', end:str='', **kw) -> DataFrame:
        '''
        field 必填 可通过类的field方法查看有哪些字段
        
        code 默认值所有的股票，['*']
        start 和 end 可以接受 YYYYMMDD/YYYYMMDDHHMMSS/YYYY-MM-DD/YYYY-MM-DD HH:MM:SS/

        函数重载
        '''
        if 'dts' not in  kw.keys():
            re = self._get_data(field, code, start, end)
            
        else:
            if field not in self.field:
                raise Exception('field 不存在')
            
            re = self.period_connect[field][kw['dts']]

        
        if field in ['l1_indus', 'l2_indus', 'l3_indus']:
            re = re.bfill().ffill()

        if self.period == '1d':
            re.index = re.index.map(lambda x: x.strftime('%Y-%m-%d'))

        return re
    
    def _get_data(self, field:str, code: list[str]=['*'], start:str='', end:str='') -> DataFrame:
        '''
        field 必填 可通过类的field方法查看有哪些字段
        
        code 默认值所有的股票，['*']
        start 和 end 可以接受 YYYYMMDD/YYYYMMDDHHMMSS/YYYY-MM-DD/YYYY-MM-DD HH:MM:SS/
        '''
        if field not in self.field:
            raise Exception('field 不存在')
        
        if not isinstance(code, list): raise Exception('code type is wrong')
        start, end = deal_start_end_time(start, end)
        dt_index = self.period_connect[field].getDateTime(start_dt=start, end_dt=end)
        if '*' in code:
            return self.period_connect[field][dt_index, :]
        else:
            return self.period_connect[field][dt_index, code]
            