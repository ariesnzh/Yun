# coding=utf-8
import os
import gc
import time
import uuid
import html
import mmap
import pickle
import datetime as dt
from collections import OrderedDict
from multiprocessing import Process, Queue, cpu_count

import sympy
import numpy as np
import pandas as pd
from progressbar import ProgressBar
from traits.api import Instance, Str, List, Int, Enum, ListStr, Either, Directory, Dict
from IPython.display import Math

# from QuantStudio import __QS_Object__, __QS_Error__, QSArgs
# from QuantStudio.FactorDataBase.FactorCache import FactorCache
# from QuantStudio.Tools.api import Panel
# from QuantStudio.Tools.IDFun import testIDFilterStr
# from QuantStudio.Tools.AuxiliaryFun import startMultiProcess, partitionListMovingSampling, partitionList
# from QuantStudio.Tools.DataPreprocessingFun import fillNaByLookback
# from QuantStudio.Tools.DataTypeConversionFun import dict2html
from dstool.QuantStudio import __QS_Object__, __QS_Error__, QSArgs
from dstool.QuantStudio.FactorDataBase.FactorCache import FactorCache
from dstool.QuantStudio.Tools.api import Panel
from dstool.QuantStudio.Tools.IDFun import testIDFilterStr
from dstool.QuantStudio.Tools.AuxiliaryFun import startMultiProcess, partitionListMovingSampling, partitionList
from dstool.QuantStudio.Tools.DataPreprocessingFun import fillNaByLookback
from dstool.QuantStudio.Tools.DataTypeConversionFun import dict2html


# 因子库, 只读, 接口类
# 数据库由若干张因子表组成
# 不支持某个操作时, 方法产生错误
# 没有相关数据时, 方法返回 None
class FactorDB(__QS_Object__):
    """因子库"""
    class __QS_ArgClass__(__QS_Object__.__QS_ArgClass__):
        Name = Str("因子库", arg_type="String", label="名称", order=-100)
    @property
    def Name(self):
        return self._QSArgs.Name
    # ------------------------------数据源操作---------------------------------
    # 链接到数据库
    def connect(self):
        return self
    # 断开到数据库的链接
    def disconnect(self):
        return 0
    # 检查数据库是否可用
    def isAvailable(self):
        return True
    # -------------------------------表的操作---------------------------------
    # 表名, 返回: [表名]
    @property
    def TableNames(self):
        return []
    # 返回因子表对象
    def getTable(self, table_name, args={}):
        return None
    def __getitem__(self, table_name):
        return self.getTable(table_name)
    def _repr_html_(self):
        return f"<b>名称</b>: {html.escape(self.Name)}<br/>" + super()._repr_html_()
    
    def equals(self, other):
        if self is other: return True
        if not isinstance(other, FactorDB): return False
        if not (isinstance(other, type(self)) or isinstance(self, type(other))): return False
        if self._QSArgs != other._QSArgs: return False
        return True


# 支持写入的因子库, 接口类
class WritableFactorDB(FactorDB):
    """可写入的因子数据库"""
    # -------------------------------表的操作---------------------------------
    # 重命名表. 必须具体化
    def renameTable(self, old_table_name, new_table_name):
        raise NotImplementedError
    # 删除表. 必须具体化
    def deleteTable(self, table_name):
        raise NotImplementedError
    # 设置表的元数据. 必须具体化
    def setTableMetaData(self, table_name, key=None, value=None, meta_data=None):
        raise NotImplementedError
    # 设置因子定义代码
    # if_exists: append, update
    def setFactorDef(self, table_name, def_file, if_exists="update"):
        raise NotImplementedError
    # --------------------------------因子操作-----------------------------------
    # 对一张表的因子进行重命名. 必须具体化
    def renameFactor(self, table_name, old_factor_name, new_factor_name):
        raise NotImplementedError
    # 删除一张表中的某些因子. 必须具体化
    def deleteFactor(self, table_name, factor_names):
        raise NotImplementedError
    # 设置因子的元数据. 必须具体化
    def setFactorMetaData(self, table_name, ifactor_name, key=None, value=None, meta_data=None):
        raise NotImplementedError
    # 写入数据, if_exists: append, update. data_type: dict like, {因子名:数据类型}, 必须具体化
    def writeData(self, data, table_name, if_exists="update", data_type={}, **kwargs):
        raise NotImplementedError
    # -------------------------------数据变换------------------------------------
    # 时间平移, 沿着时间轴将所有数据纵向移动 lag 期, lag>0 向前移动, lag<0 向后移动, 空出来的地方填 nan
    def offsetDateTime(self, lag, table_name, factor_names, args={}):
        if lag==0: return 0
        FT = self.getTable(table_name, args=args)
        Data = FT.readData(factor_names=factor_names, ids=FT.getID(), dts=FT.getDateTime(), args=args)
        if lag>0:
            Data.iloc[:, lag:, :] = Data.iloc[:,:-lag,:].values
            Data.iloc[:, :lag, :] = None
        elif lag<0:
            Data.iloc[:, :lag, :] = Data.iloc[:,-lag:,:].values
            Data.iloc[:, :lag, :] = None
        DataType = FT.getFactorMetaData(factor_names, key="DataType", args=args).to_dict()
        self.deleteFactor(table_name, factor_names)
        self.writeData(Data, table_name, data_type=DataType)
        return 0
    # 数据变换, 对原来的时间和ID序列通过某种变换函数得到新的时间序列和ID序列, 调整数据
    def changeData(self, table_name, factor_names, ids, dts, args={}):
        FT = self.getTable(table_name, args=args)
        Data = FT.readData(factor_names=factor_names, ids=ids, dts=dts, args=args)
        DataType = FT.getFactorMetaData(factor_names, key="DataType", args=args).to_dict()
        self.deleteFactor(table_name, factor_names)
        self.writeData(Data, table_name, data_type=DataType)
        return 0
    # 填充缺失值
    def fillNA(self, filled_value, table_name, factor_names, ids, dts, args={}):
        Data = self.getTable(table_name).readData(factor_names=factor_names, ids=ids, dts=dts, args=args)
        Data.fillna(filled_value, inplace=True)
        self.writeData(Data, table_name, if_exists='update')
        return 0
    # 替换数据
    def replaceData(self, old_value, new_value, table_name, factor_names, ids, dts, args={}):
        Data = self.getTable(table_name).readData(factor_names=factor_names, ids=ids, dts=dts, args=args)
        Data = Data.where(Data!=old_value, new_value)
        self.writeData(Data, table_name, if_exists='update')
        return 0
    # 优化数据
    def optimizeData(self, table_name, factor_names):
        raise NotImplementedError
    # 修复数据, 依赖具体实现, 不保证一定修复
    def fixData(self, table_name, factor_names):
        raise NotImplementedError

# 因子表的遍历模式参数对象
class _ErgodicMode(QSArgs):
    """遍历模式"""
    ForwardPeriod = Int(600, arg_type="Integer", label="向前缓冲时点数", order=0)
    BackwardPeriod = Int(1, arg_type="Integer", label="向后缓冲时点数", order=1)
    CacheMode = Enum("因子", "ID", arg_type="SingleOption", label="缓冲模式", order=2, option_range=("因子", "ID"))
    MaxFactorCacheNum = Int(60, arg_type="Integer", label="最大缓冲因子数", order=3)
    MaxIDCacheNum = Int(10000, arg_type="Integer", label="最大缓冲ID数", order=4)
    CacheSize = Int(300, arg_type="Integer", label="缓冲区大小", order=5)# 以 MB 为单位
    ErgodicDTs = List(arg_type="DateTimeList", label="遍历时点", order=6)
    ErgodicIDs = List(arg_type="IDList", label="遍历ID", order=7)
    AutoMove = Enum(True, False, label="自动缓冲", arg_type="Bool", order=8)
    CacheDir = Directory(arg_type="Directory", label="缓存文件夹", order=9)
    ClearCache = Enum(True, False, arg_type="Bool", label="清空缓存", order=10)
    def __init__(self, owner=None, sys_args={}, config_file=None, **kwargs):
        super().__init__(owner=owner, sys_args=sys_args, config_file=config_file, **kwargs)
        self._isStarted = False
        self._CurDT = None
        self._FactorCache = None

    def __getstate__(self):
        state = self.__dict__.copy()
        if "_CacheDataProcess" in state: state["_CacheDataProcess"] = None
        return state

    # 因子缓存对象
    @property
    def FactorCache(self):
        return self._FactorCache

    # 启动遍历模式, dts: 遍历的时间点序列或者迭代器
    def start(self, dts, **kwargs):
        if self._isStarted: return 0
        self._DateTimes = np.array((self._Owner.getDateTime() if not self.ErgodicDTs else self.ErgodicDTs), dtype="O")
        if self._DateTimes.shape[0]==0: raise __QS_Error__("因子表: '%s' 的默认时间序列为空, 请设置参数 '遍历模式-遍历时点' !" % self._Owner.Name)
        self._IDs = (self._Owner.getID() if not self.ErgodicIDs else list(self.ErgodicIDs))
        if not self._IDs: raise __QS_Error__("因子表: '%s' 的默认 ID 序列为空, 请设置参数 '遍历模式-遍历ID' !" % self._Owner.Name)
        self._CurInd = -1# 当前时点在 dts 中的位置, 以此作为缓冲数据的依据
        self._DTNum = self._DateTimes.shape[0]# 时点数
        self._CacheDTs = []# 缓冲的时点序列
        self._CacheData = {}# 当前缓冲区
        self._CacheFactorNum = 0# 当前缓存因子个数, 小于等于 self.MaxFactorCacheNum
        self._CacheIDNum = 0# 当前缓存ID个数, 小于等于 self.MaxIDCacheNum
        self._FactorReadNum = pd.Series(0, index=self._Owner.FactorNames)# 因子读取次数, pd.Series(读取次数, index=self._Owner.FactorNames)
        self._IDReadNum = pd.Series()# ID读取次数, pd.Series(读取次数, index=self._Owner.FactorNames)
        self._Queue2SubProcess = Queue()# 主进程向数据准备子进程发送消息的管道
        self._Queue2MainProcess = Queue()# 数据准备子进程向主进程发送消息的管道
        if self.CacheSize>0:
            if os.name=="nt":
                self._TagName = str(uuid.uuid1())# 共享内存的 tag
                self._MMAPCacheData = None
            else:
                self._TagName = None# 共享内存的 tag
                self._MMAPCacheData = mmap.mmap(-1, int(self.CacheSize*2**20))# 当前共享内存缓冲区
            if self.CacheMode=="因子": self._CacheDataProcess = Process(target=_prepareMMAPFactorCacheData, args=(self._Owner, self._MMAPCacheData), daemon=True)
            else: self._CacheDataProcess = Process(target=_prepareMMAPIDCacheData, args=(self._Owner, self._MMAPCacheData), daemon=True)
            self._CacheDataProcess.start()
            if os.name=="nt": self._MMAPCacheData = mmap.mmap(-1, int(self.CacheSize*2**20), tagname=self._TagName)# 当前共享内存缓冲区
        if self.CacheDir and os.path.isdir(self.CacheDir):
            self._FactorCache = FactorCache(sys_args={"缓存文件夹": self.CacheDir, "进程ID": {"0-0": self._IDs}})
        self._isStarted = True
        return 0
    
    # 时间点向前移动, idt: 时间点, datetime.dateime
    def move(self, idt, **kwargs):
        if idt==self._CurDT: return 0
        self._CurDT = idt
        PreInd = self._CurInd
        self._CurInd = PreInd + np.sum(self._DateTimes[PreInd+1:]<=idt)
        if (self.CacheSize>0) and (self._CurInd>-1) and ((not self._CacheDTs) or (self._DateTimes[self._CurInd]>self._CacheDTs[-1])):# 需要读入缓冲区的数据
            self._Queue2SubProcess.put((None, None))
            DataLen = self._Queue2MainProcess.get()
            CacheData = b""
            while DataLen>0:
                self._MMAPCacheData.seek(0)
                CacheData += self._MMAPCacheData.read(DataLen)
                self._Queue2SubProcess.put(DataLen)
                DataLen = self._Queue2MainProcess.get()
            self._CacheData = pickle.loads(CacheData)
            if self._CurInd==PreInd+1:# 没有跳跃, 连续型遍历
                self._Queue2SubProcess.put((self._CurInd, None))
                self._CacheDTs = self._DateTimes[max((0, self._CurInd-self.BackwardPeriod)):min((self._DTNum, self._CurInd+self.ForwardPeriod+1))].tolist()
            else:# 出现了跳跃
                LastCacheInd = (self._DateTimes.searchsorted(self._CacheDTs[-1]) if self._CacheDTs else self._CurInd-1)
                self._Queue2SubProcess.put((LastCacheInd+1, None))
                self._CacheDTs = self._DateTimes[max((0, LastCacheInd+1-self.BackwardPeriod)):min((self._DTNum, LastCacheInd+1+self.ForwardPeriod+1))].tolist()
        return 0
    
    # 结束遍历模式
    def end(self):
        if not self._isStarted: return 0
        self._CacheData, self._FactorReadNum, self._IDReadNum = None, None, None
        if self.CacheSize>0: self._Queue2SubProcess.put(None)
        self._Queue2SubProcess = self._Queue2MainProcess = self._CacheDataProcess = None
        self._isStarted = False
        self._CurDT = None
        self._MMAPCacheData = None
        if self.ClearCache: self._FactorCache.clear()
        return 0

    def _readData_FactorCacheMode(self, factor_names, ids, dts, args={}):
        FT = self._Owner
        self._FactorReadNum[factor_names] += 1
        if (self.MaxFactorCacheNum<=0) or (not self._CacheDTs) or (dts[0]<self._CacheDTs[0]) or (dts[-1]>self._CacheDTs[-1]):
            #print("超出缓存区读取: "+str(factor_names))# debug
            return FT.__QS_calcData__(raw_data=FT.__QS_prepareRawData__(factor_names=factor_names, ids=ids, dts=dts, args=args), factor_names=factor_names, ids=ids, dts=dts, args=args)
        Data = {}
        DataFactorNames = []
        CacheFactorNames = set()
        PopFactorNames = []
        for iFactorName in factor_names:
            iFactorData = self._CacheData.get(iFactorName)
            if iFactorData is None:# 尚未进入缓存
                if self._CacheFactorNum<self.MaxFactorCacheNum:# 当前缓存因子数小于最大缓存因子数，那么将该因子数据读入缓存
                    self._CacheFactorNum += 1
                    CacheFactorNames.add(iFactorName)
                else:# 当前缓存因子数等于最大缓存因子数，那么将检查最小读取次数的因子
                    CacheFactorReadNum = self._FactorReadNum[self._CacheData.keys()]
                    MinReadNumInd = CacheFactorReadNum.argmin()
                    if CacheFactorReadNum.loc[MinReadNumInd]<self._FactorReadNum[iFactorName]:# 当前读取的因子的读取次数超过了缓存因子读取次数的最小值，缓存该因子数据
                        CacheFactorNames.add(iFactorName)
                        PopFactor = MinReadNumInd
                        self._CacheData.pop(PopFactor)
                        PopFactorNames.append(PopFactor)
                    else:
                        DataFactorNames.append(iFactorName)
            else:
                Data[iFactorName] = iFactorData
        CacheFactorNames = list(CacheFactorNames)
        if CacheFactorNames:
            #print("尚未进入缓存区读取: "+str(CacheFactorNames))# debug
            iData = dict(FT.__QS_calcData__(raw_data=FT.__QS_prepareRawData__(factor_names=CacheFactorNames, ids=self._IDs, dts=self._CacheDTs, args=args), factor_names=CacheFactorNames, ids=self._IDs, dts=self._CacheDTs, args=args))
            Data.update(iData)
            self._CacheData.update(iData)
        self._Queue2SubProcess.put((None, (CacheFactorNames, PopFactorNames)))
        Data = Panel(Data)
        if Data.shape[0]>0:
            try:
                Data = Data.loc[:, dts, ids]
            except KeyError as e:
                self._QS_Logger.warning("FactorTable._readData_FactorCacheMode : %s 提取的时点或 ID 不在因子表范围内: %s" % (FT.Name, str(e)))
                Data = Panel(items=Data.items, major_axis=dts, minor_axis=ids)
        if not DataFactorNames: return Data.loc[factor_names]
        #print("超出缓存区因子个数读取: "+str(DataFactorNames))# debug
        return FT.__QS_calcData__(raw_data=FT.__QS_prepareRawData__(factor_names=DataFactorNames, ids=ids, dts=dts, args=args), factor_names=DataFactorNames, ids=ids, dts=dts, args=args).join(Data).loc[factor_names]
    
    def _readIDData(self, iid, factor_names, dts, args={}):
        FT = self._Owner
        self._IDReadNum[iid] = self._IDReadNum.get(iid, 0) + 1
        if (self.MaxIDCacheNum<=0) or (not self._CacheDTs) or (dts[0] < self._CacheDTs[0]) or (dts[-1] >self._CacheDTs[-1]):
            return FT.__QS_calcData__(raw_data=FT.__QS_prepareRawData__(factor_names=factor_names, ids=[iid], dts=dts, args=args), factor_names=factor_names, ids=[iid], dts=dts, args=args).iloc[:, :, 0]
        IDData = self._CacheData.get(iid)
        if IDData is None:# 尚未进入缓存
            if self._CacheIDNum<self.MaxIDCacheNum:# 当前缓存 ID 数小于最大缓存 ID 数，那么将该 ID 数据读入缓存
                self._CacheIDNum += 1
                IDData = FT.__QS_calcData__(raw_data=FT.__QS_prepareRawData__(factor_names=self.FactorNames, ids=[iid], dts=self._CacheDTs, args=args), factor_names=self.FactorNames, ids=[iid], dts=self._CacheDTs, args=args).iloc[:, :, 0]
                self._CacheData[iid] = IDData
                self._Queue2SubProcess.put((None, (iid, None)))
            else:# 当前缓存 ID 数等于最大缓存 ID 数，那么将检查最小读取次数的 ID
                CacheIDReadNum = self._IDReadNum[self._CacheData.keys()]
                MinReadNumInd = CacheIDReadNum.argmin()
                if CacheIDReadNum.loc[MinReadNumInd]<self._IDReadNum[iid]:# 当前读取的 ID 的读取次数超过了缓存 ID 读取次数的最小值，缓存该 ID 数据
                    IDData = FT.__QS_calcData__(raw_data=FT.__QS_prepareRawData__(factor_names=self.FactorNames, ids=[iid], dts=self._CacheDTs, args=args), factor_names=self.FactorNames, ids=[iid], dts=self._CacheDTs, args=args).iloc[:, :, 0]
                    PopID = MinReadNumInd
                    self._CacheData.pop(PopID)
                    self._CacheData[iid] = IDData
                    self._Queue2SubProcess.put((None, (iid, PopID)))
                else:# 当前读取的 ID 的读取次数没有超过缓存 ID 读取次数的最小值, 放弃缓存该 ID 数据
                    return FT.__QS_calcData__(raw_data=FT.__QS_prepareRawData__(factor_names=factor_names, ids=[iid], dts=dts, args=args), factor_names=factor_names, ids=[iid], dts=dts, args=args).iloc[:, :, 0]
        return IDData.reindex(index=dts, columns=factor_names)
    
    def readData(self, factor_names, ids, dts, args={}):
        if args.get("遍历模式", {}).get("自动缓冲", self.AutoMove) and dts:
            self.move(dts[-1])
        if self.CacheMode=="因子":
            return self._readData_FactorCacheMode(factor_names=factor_names, ids=ids, dts=dts, args=args)
        else:
            return Panel({iID: self._readIDData(iID, factor_names=factor_names, dts=dts, args=args) for iID in ids}, items=ids, major_axis=dts, minor_axis=factor_names).swapaxes(0, 2)

# 基于 mmap 的缓冲数据, 如果开启遍历模式, 那么限制缓冲的因子个数, ID 个数, 时间点长度, 缓冲区里是因子的部分数据
def _prepareMMAPFactorCacheData(ft, mmap_cache):
    ErgodicMode = ft._QSArgs.ErgodicMode
    CacheData, CacheDTs, MMAPCacheData, DTNum = {}, [], mmap_cache, len(ErgodicMode._DateTimes)
    CacheSize = int(ErgodicMode.CacheSize*2**20)
    if os.name=='nt': MMAPCacheData = mmap.mmap(-1, CacheSize, tagname=ErgodicMode._TagName)
    while True:
        Task = ErgodicMode._Queue2SubProcess.get()# 获取任务
        if Task is None: break# 结束进程
        if (Task[0] is None) and (Task[1] is None):# 把数据装入缓存区
            CacheDataByte = pickle.dumps(CacheData)
            DataLen = len(CacheDataByte)
            for i in range(int(DataLen/CacheSize)+1):
                iStartInd = i*CacheSize
                iEndInd = min((i+1)*CacheSize, DataLen)
                if iEndInd>iStartInd:
                    MMAPCacheData.seek(0)
                    MMAPCacheData.write(CacheDataByte[iStartInd:iEndInd])
                    ErgodicMode._Queue2MainProcess.put(iEndInd-iStartInd)
                    ErgodicMode._Queue2SubProcess.get()
            ErgodicMode._Queue2MainProcess.put(0)
            del CacheDataByte
            gc.collect()
        elif Task[0] is None:# 调整缓存区
            NewFactors, PopFactors = Task[1]
            for iFactorName in PopFactors: CacheData.pop(iFactorName)
            if NewFactors:
                #print("调整缓存区: "+str(NewFactors))# debug
                if CacheDTs:
                    CacheData.update(dict(ft.__QS_calcData__(raw_data=ft.__QS_prepareRawData__(factor_names=NewFactors, ids=ErgodicMode._IDs, dts=CacheDTs), factor_names=NewFactors, ids=ErgodicMode._IDs, dts=CacheDTs)))
                else:
                    CacheData.update({iFactorName: pd.DataFrame(index=CacheDTs, columns=ErgodicMode._IDs) for iFactorName in NewFactors})
        else:# 准备缓存区
            CurInd = Task[0] + ErgodicMode.ForwardPeriod + 1
            if CurInd < DTNum:# 未到结尾处, 需要再准备缓存数据
                OldCacheDTs = set(CacheDTs)
                CacheDTs = ErgodicMode._DateTimes[max((0, CurInd-ErgodicMode.BackwardPeriod)):min((DTNum, CurInd+ErgodicMode.ForwardPeriod+1))].tolist()
                NewCacheDTs = sorted(set(CacheDTs).difference(OldCacheDTs))
                if CacheData:
                    isDisjoint = OldCacheDTs.isdisjoint(CacheDTs)
                    CacheFactorNames = list(CacheData.keys())
                    #print("准备缓存区: "+str(CacheFactorNames))# debug
                    if NewCacheDTs:
                        NewCacheData = ft.__QS_calcData__(raw_data=ft.__QS_prepareRawData__(factor_names=CacheFactorNames, ids=ErgodicMode._IDs, dts=NewCacheDTs), factor_names=CacheFactorNames, ids=ErgodicMode._IDs, dts=NewCacheDTs)
                    else:
                        NewCacheData = Panel(items=CacheFactorNames, major_axis=NewCacheDTs, minor_axis=ErgodicMode._IDs)
                    for iFactorName in CacheData:
                        if isDisjoint:
                            CacheData[iFactorName] = NewCacheData[iFactorName]
                        else:
                            CacheData[iFactorName] = CacheData[iFactorName].reindex(index=CacheDTs)
                            CacheData[iFactorName].loc[NewCacheDTs, :] = NewCacheData[iFactorName]
                    NewCacheData = None
    return 0
# 基于 mmap 的 ID 缓冲的因子表, 如果开启遍历模式, 那么限制缓冲的 ID 个数和时间点长度, 缓冲区里是 ID 的部分数据
def _prepareMMAPIDCacheData(ft, mmap_cache):
    ErgodicMode = ft._QSArgs.ErgodicMode
    CacheData, CacheDTs, MMAPCacheData, DTNum = {}, [], mmap_cache, len(ErgodicMode._DateTimes)
    CacheSize = int(ErgodicMode.CacheSize*2**20)
    if os.name=='nt': MMAPCacheData = mmap.mmap(-1, CacheSize, tagname=ErgodicMode._TagName)
    while True:
        Task = ErgodicMode._Queue2SubProcess.get()# 获取任务
        if Task is None: break# 结束进程
        if (Task[0] is None) and (Task[1] is None):# 把数据装入缓冲区
            CacheDataByte = pickle.dumps(CacheData)
            DataLen = len(CacheDataByte)
            for i in range(int(DataLen/CacheSize)+1):
                iStartInd = i*CacheSize
                iEndInd = min((i+1)*CacheSize, DataLen)
                if iEndInd>iStartInd:
                    MMAPCacheData.seek(0)
                    MMAPCacheData.write(CacheDataByte[iStartInd:iEndInd])
                    ErgodicMode._Queue2MainProcess.put(iEndInd-iStartInd)
                    ErgodicMode._Queue2SubProcess.get()
            ErgodicMode._Queue2MainProcess.put(0)
            del CacheDataByte
            gc.collect()
        elif Task[0] is None:# 调整缓存区数据
            NewID, PopID = Task[1]
            if PopID: CacheData.pop(PopID)# 用新 ID 数据替换旧 ID
            if NewID:
                if CacheDTs:
                    CacheData[NewID] = ft.__QS_calcData__(raw_data=ft.__QS_prepareRawData__(factor_names=ft.FactorNames, ids=[NewID], dts=CacheDTs), factor_names=ft.FactorNames, ids=[NewID], dts=CacheDTs).iloc[:, :, 0]
                else:
                    CacheData[NewID] = pd.DataFrame(index=CacheDTs, columns=ft.FactorNames)
        else:# 准备缓冲区
            CurInd = Task[0] + ErgodicMode.ForwardPeriod + 1
            if CurInd<DTNum:# 未到结尾处, 需要再准备缓存数据
                OldCacheDTs = set(CacheDTs)
                CacheDTs = ErgodicMode._DateTimes[max((0, CurInd-ErgodicMode.BackwardPeriod)):min((DTNum, CurInd+ErgodicMode.ForwardPeriod+1))].tolist()
                NewCacheDTs = sorted(set(CacheDTs).difference(OldCacheDTs))
                if CacheData:
                    isDisjoint = OldCacheDTs.isdisjoint(CacheDTs)
                    CacheIDs = list(CacheData.keys())
                    if NewCacheDTs:
                        NewCacheData = ft.__QS_calcData__(raw_data=ft.__QS_prepareRawData__(factor_names=ft.FactorNames, ids=CacheIDs, dts=NewCacheDTs), factor_names=ft.FactorNames, ids=CacheIDs, dts=NewCacheDTs)
                    else:
                        NewCacheData = Panel(items=ft.FactorNames, major_axis=NewCacheDTs, minor_axis=CacheIDs)
                    for iID in CacheData:
                        if isDisjoint:
                            CacheData[iID] = NewCacheData.loc[:, :, iID]
                        else:
                            CacheData[iID] = CacheData[iID].reindex(index=CacheDTs)
                            CacheData[iID].loc[NewCacheDTs, :] = NewCacheData.loc[:, :, iID]
                    NewCacheData = None
    return 0

# 因子表的批量模式参数对象
class _OperationMode(QSArgs):
    """批量模式"""
    DateTimes = List(dt.datetime, arg_type="DateTimeList", label="运算时点", order=0)
    IDs = ListStr(arg_type="IDList", label="运算ID", order=1)
    FactorNames = ListStr(arg_type="MultiOption", label="运算因子", order=2)
    SubProcessNum = Int(0, arg_type="Integer", label="子进程数", order=3)
    DTRuler = List(dt.datetime, arg_type="DateTimeList", label="时点标尺", order=4)
    SectionIDs = Either(None, ListStr, arg_type="IDList", label="截面ID", order=5)
    IDSplit = Enum("连续切分", "间隔切分", arg_type="SingleOption", label="ID切分", order=6)
    CacheDir = Directory(arg_type="Directory", label="缓存文件夹", order=7)
    ClearCache = Enum(True, False, arg_type="Bool", label="清空缓存", order=8)
    WriteBatchNum = Int(1, arg_type="Integer", label="写入批次", order=9)
    def __init__(self, owner=None, sys_args={}, config_file=None, **kwargs):
        self._FT = owner
        self._isStarted = False
        self._Factors = []# 因子列表, 只包含当前生成数据的因子
        self._FactorDict = {}# 因子字典, {因子名:因子}, 包括所有的因子, 即衍生因子所依赖的描述子也在内
        self._FactorID = {}# {因子名: 因子唯一的 ID 号(int)}, 比如防止操作系统文件大小写不敏感导致缓存文件重名
        self._Factor2RawFactor = {}  # 因子对应的基础因子名称列表, {因子名: {基础因子名}}
        self._FactorStartDT = {}# {因子名: 起始时点}
        self._FactorPrepareIDs = {}# {因子名: 需要准备原始数据的 ID 序列}
        self._iPID = "0"# 对象所在的进程 ID
        self._PIDs = []# 所有的计算进程 ID, 单进程下默认为"0", 多进程为"0-i"
        self._PID_IDs = {}# 每个计算进程分配的 ID 列表, {PID:[ID]}
        self._PID_Lock = {}# 每个计算进程分配的缓存数据锁, {PID:Lock}
        self._Cache = None# 因子缓存对象
        self._Event = {}# {因子名: (Sub2MainQueue, Event)}
        self._FileSuffix = ".h5"
        super().__init__(owner=owner, sys_args=sys_args, config_file=config_file, **kwargs)
    
    def __QS_initArgs__(self, args={}):
        self.add_trait("FactorNames", ListStr(arg_type="MultiOption", label="运算因子", order=2))
    
    def _genFactorDict(self, factors, factor_dict, parent_factor=None):
        for iFactor in factors:
            iFactor._OperationMode = self
            if (not isinstance(iFactor.Name, str)) or (iFactor.Name=="") or (iFactor is not factor_dict.get(iFactor.Name, iFactor)):# 该因子命名错误或者未命名, 或者有因子重名
                # iNewName = genAvailableName("TempFactor", factor_dict)
                iNewName = f"_QS_TempFactor_{id(iFactor)}"
                iFactor.Name, self._FactorNameChgRecord[iNewName] = iNewName, iFactor.Name
            factor_dict[iFactor.Name] = iFactor
            self._FactorID[iFactor.Name] = id(iFactor)# len(factor_dict)
            iParentFactor = (iFactor if parent_factor is None else parent_factor)
            factor_dict.update(self._genFactorDict(iFactor.Descriptors, factor_dict, iParentFactor))
            if iFactor.FactorTable is not None:
                self._Factor2RawFactor[iParentFactor.Name].add(iFactor.Name)
        return factor_dict

    # 初始化模式, 只在模式开始时运行
    def _initMode(self, **kwargs):
        self._isStarted = True
        self._Factor2RawFactor = {}# 因子对应的基础因子名称列表, {因子名: {基础因子名}}
        self._Factors = []  # 因子列表, 只包括需要输出数据的因子对象
        self._FactorDict = {}  # 因子字典, {因子名:因子}, 包括所有的因子, 即衍生因子所依赖的描述子也在内
        self._FactorID = {}  # {因子名: 因子唯一的 ID 号(int)}
        self._FactorNameChgRecord = {}  # 因子名的修改记录, {修改后的名字: 原始名字}
        for i, iFactorName in enumerate(self._Owner.FactorNames):
            iFactor = self._Owner.getFactor(iFactorName)
            iFactor._OperationMode = self
            self._Factors.append(iFactor)
            self._FactorDict[iFactorName] = iFactor
            self._FactorID[iFactorName] = id(iFactor)# i
            self._Factor2RawFactor[iFactorName] = set()
        self._FactorDict = self._genFactorDict(self._Factors, self._FactorDict)
        # 分配每个子进程的计算 ID 序列, 生成原始数据和缓存数据存储目录
        if not self.SectionIDs: self.SectionIDs = self._FT.getID()
        if self.SubProcessNum == 0:# 串行模式
            self._PIDs = ["0"]
            self._PID_IDs = {"0": list(self.SectionIDs)}
        else:
            self._PIDs = []
            self._PID_IDs = {}
            nPrcs = min((self.SubProcessNum, len(self.SectionIDs)))
            if self.IDSplit == "连续切分":
                SubIDs = partitionList(list(self.SectionIDs), nPrcs)
            elif self.IDSplit == "间隔切分":
                SubIDs = partitionListMovingSampling(list(self.SectionIDs), nPrcs)
            else:
                raise __QS_Error__(f"不支持的 ID 切分方式: {self.IDSplit}")
            for i in range(nPrcs):
                iPID = "0-" + str(i)
                self._PIDs.append(iPID)
                self._PID_IDs[iPID] = SubIDs[i]
        self._Cache = FactorCache(sys_args={"缓存文件夹": kwargs.get("cache_dir", self.CacheDir), "进程ID": self._PID_IDs})
        # 遍历所有因子对象, 调用其初始化方法, 生成所有因子的起始时点信息, 生成其需要准备原始数据的截面 ID
        self._FactorStartDT = {}# {因子名: 起始时点}
        self._FactorPrepareIDs = {}# {因子名: 需要准备原始数据的 ID 序列}
        for iFactor in self._Factors:
            iFactor._QS_initOperation(self.DateTimes[0], self._FactorStartDT, self.SectionIDs, self._FactorPrepareIDs)
        # 分组准备数据
        InitGroups = {}  # {id(因子表) : [(因子表, [因子], [ID])]}
        for iFactor in self._FactorDict.values():
            if iFactor.FactorTable is None: continue
            iFTID = id(iFactor.FactorTable)
            iPrepareIDs = self._FactorPrepareIDs[iFactor.Name]
            if iFTID not in InitGroups:
                InitGroups[iFTID] = [(iFactor.FactorTable, [iFactor], iPrepareIDs)]
            else:
                iGroups = InitGroups[iFTID]
                for j in range(len(iGroups)):
                    if iPrepareIDs == iGroups[j][2]:
                        iGroups[j][1].append(iFactor)
                        break
                else:
                    iGroups.append((iFactor.FactorTable, [iFactor], iPrepareIDs))
        self._RawFactorGroupIdx = {}# {基础因子名: Int}
        GroupInfo, RawDataFileNames, PrepareIDs, PID_PrepareIDs = [], [], [], []#[(因子表对象, [因子名], [原始因子名], [时点], {参数})], [原始数据文件名], [准备数据的ID序列]
        for iFTID, iGroups in InitGroups.items():
            iGroupInfo = []
            jStartInd = 0
            for j in range(len(iGroups)):
                iFT = iGroups[j][0]
                ijGroupInfo = iFT.__QS_genGroupInfo__(iGroups[j][1], self)
                iGroupInfo.extend(ijGroupInfo)
                ijGroupNum = len(ijGroupInfo)
                for k in range(ijGroupNum):
                    ijkRawDataFileName = iFT.Name+"-"+str(iFTID)+"-"+str(jStartInd+k)
                    for m in range(len(ijGroupInfo[k][1])):
                        self._FactorDict[ijGroupInfo[k][1][m]]._RawDataFile = ijkRawDataFileName
                        self._RawFactorGroupIdx[ijGroupInfo[k][1][m]] = len(GroupInfo) + jStartInd + k
                    RawDataFileNames.append(ijkRawDataFileName)
                jStartInd += ijGroupNum
                PrepareIDs += [iGroups[j][2]] * ijGroupNum
                if iGroups[j][2] is not None:
                    PID_PrepareIDs += [{self._PIDs[i]: iSubIDs for i, iSubIDs in enumerate(partitionListMovingSampling(iGroups[j][2], len(self._PIDs)))}] * ijGroupNum
                else:
                    PID_PrepareIDs += [None] * ijGroupNum
            GroupInfo.extend(iGroupInfo)
        self._RawFactorGroupIdx = pd.Series(self._RawFactorGroupIdx, dtype=int)
        self._RawDataPreparation = dict(GroupInfo=GroupInfo, RawDataFileNames=RawDataFileNames, PrepareIDs=PrepareIDs, PID_PrepareIDs=PID_PrepareIDs)

    def _initOperation(self, **kwargs):
        # 检查时点, ID 序列的合法性
        if not self.DateTimes: raise __QS_Error__("运算时点序列不能为空!")
        if not self.IDs: raise __QS_Error__("运算 ID 序列不能为空!")
        # 检查时点标尺是否合适
        DTs = pd.Series(np.arange(0, len(self.DTRuler)), index=list(self.DTRuler)).reindex(index=list(self.DateTimes))
        if pd.isnull(DTs).any(): raise __QS_Error__("运算时点序列超出了时点标尺!")
        elif (DTs.diff().iloc[1:]!=1).any(): raise __QS_Error__("运算时点序列的频率与时点标尺不一致!")
        # 检查因子的合法性, 解析出所有的因子(衍生因子所依赖的描述子也在内)
        if not self.FactorNames: self.FactorNames = self._Owner.FactorNames
        self._Event = {}# {因子名: (Sub2MainQueue, Event)}, 用于多进程同步的 Event 数据

    # TODO: 重用已经有的原始数据缓存
    def _prepare(self, **kwargs):
        GroupIdx = []
        for iFactorName in self.FactorNames:
            GroupIdx += self._RawFactorGroupIdx.loc[list(self._Factor2RawFactor[iFactorName])].tolist()
        args = {"FT": self._Owner, "GroupInfo": [], "RawDataFileNames": [], "PrepareIDs": [], "PID_PrepareIDs": []}
        for i in sorted(set(GroupIdx)):
            args["GroupInfo"].append(self._RawDataPreparation["GroupInfo"][i])
            args["RawDataFileNames"].append(self._RawDataPreparation["RawDataFileNames"][i])
            args["PrepareIDs"].append(self._RawDataPreparation["PrepareIDs"][i])
            args["PID_PrepareIDs"].append(self._RawDataPreparation["PID_PrepareIDs"][i])
        if self.SubProcessNum==0:
            Error = _prepareRawData(args)
        else:
            nPrcs = min((self.SubProcessNum, len(args["GroupInfo"])))
            Procs,Main2SubQueue,Sub2MainQueue = startMultiProcess(pid="0", n_prc=nPrcs, target_fun=_prepareRawData,
                                                                  arg=args, partition_arg=["GroupInfo", "RawDataFileNames", "PrepareIDs", "PID_PrepareIDs"],
                                                                  n_partition_head=0, n_partition_tail=0,
                                                                  main2sub_queue="None", sub2main_queue="Single")
            nGroup = len(GroupInfo)
            with ProgressBar(max_value=nGroup) as ProgBar:
                for i in range(nGroup):
                    iPID, Error, iMsg = Sub2MainQueue.get()
                    if Error!=1:
                        for iPID, iProc in Procs.items():
                            if iProc.is_alive(): iProc.terminate()
                        raise __QS_Error__(iMsg)
                    ProgBar.update(i+1)
            for iPrcs in Procs.values(): iPrcs.join()
        return 0
    def _exit(self):
        if self.ClearCache: self._Cache.clear()
        for iFactorName, iFactor in self._FactorDict.items():
            iFactor._exit()
            iFactor.Name = self._FactorNameChgRecord.get(iFactor.Name, iFactor.Name)
        return 0

    def _calculate(self, factor_db, table_name, if_exists, specific_target, **kwargs):
        Args = {"FT": self._Owner, "PID": "0", "FactorDB": factor_db, "TableName": table_name, "if_exists": if_exists, "specific_target": specific_target, "kwargs": kwargs}
        if self.SubProcessNum == 0:
            _calculate(Args)
        else:
            nPrcs = len(self._PIDs)
            nTask = len(self._Factors) * nPrcs
            EventState = {iFactorName: 0 for iFactorName in self._Event if iFactorName in self.FactorNames}
            Procs, Main2SubQueue, Sub2MainQueue = startMultiProcess(pid="0", n_prc=nPrcs, target_fun=_calculate, arg=Args, main2sub_queue="None", sub2main_queue="Single")
            iProg = 0
            with ProgressBar(max_value=nTask) as ProgBar:
                while True:
                    nEvent = len(EventState)
                    if nEvent > 0:
                        FactorNames = tuple(EventState.keys())
                        for iFactorName in FactorNames:
                            iQueue = self._Event[iFactorName][0]
                            while not iQueue.empty():
                                jInc = iQueue.get()
                                EventState[iFactorName] += jInc
                            if EventState[iFactorName] >= nPrcs:
                                self._Event[iFactorName][1].set()
                                EventState.pop(iFactorName)
                    while ((not Sub2MainQueue.empty()) or (nEvent == 0)) and (iProg < nTask):
                        iPID, iSubProg, iMsg = Sub2MainQueue.get()
                        iProg += iSubProg
                        ProgBar.update(iProg)
                    if iProg >= nTask: break
            for iPID, iPrcs in Procs.items(): iPrcs.join()
        return 0

    def write2FDB(self, factor_names, ids, dts, factor_db, table_name, if_exists="update", specific_target={}, args={}, **kwargs):
        if not isinstance(factor_db, WritableFactorDB): raise __QS_Error__("因子数据库: %s 不可写入!" % factor_db.Name)
        OldArgs = self.to_dict()
        print("==========因子运算==========\n1. 原始数据准备\n")
        TotalStartT = time.perf_counter()
        self.SubProcessNum = args.get("子进程数", self.SubProcessNum)
        DTRuler = args.get("时点标尺", self.DTRuler)
        self.DTRuler = (dts if DTRuler is None else DTRuler)
        self.SectionIDs = args.get("截面ID", self.SectionIDs)
        self.FactorNames = factor_names
        self.DateTimes = dts
        self.IDs = ids
        self._initMode(**kwargs)
        self._initOperation(**kwargs)
        self._prepare(**kwargs)
        print(("耗时 : %.2f" % (time.perf_counter()-TotalStartT, ))+"\n2. 因子数据计算\n")
        StartT = time.perf_counter()
        self._calculate(factor_db, table_name, if_exists, specific_target, **kwargs)
        print(("耗时 : %.2f" % (time.perf_counter()-StartT, ))+"\n3. 清理缓存\n")
        StartT = time.perf_counter()
        factor_db.connect()
        self._exit()
        self._isStarted = False
        print(('耗时 : %.2f' % (time.perf_counter()-StartT, ))+"\n"+("总耗时 : %.2f" % (time.perf_counter()-TotalStartT, ))+"\n"+"="*28)
        self.update(OldArgs)
        return 0
    
    def readData(self, factor_names, ids, dts, args={}):
        from QuantStudio.FactorDataBase.MemoryDB import MemoryDB
        MDB = MemoryDB().connect()
        if (dts is not None) and (set(dts).difference(self.DateTimes)):
            self._QS_Logger.warning(f"时点序列超出了批量模式启动时预设的时点序列, 超出部分将置为 None!")
        if (ids is not None) and (set(ids).difference(self.SectionIDs)):
            self._QS_Logger.warning(f"ID序列超出了批量模式启动时预设的截面ID序列, 超出部分将置为 None!")
        self.FactorNames = factor_names
        self._initOperation()
        self._prepare()
        self._calculate(MDB, "tmp_table", if_exists="update", specific_target={})
        # self._exit()
        return MDB["tmp_table"].readData(factor_names=factor_names, ids=ids, dts=dts)

    # 启动批量模式
    def start(self, dts, ids=None, **kwargs):
        if self._isStarted: return 0
        self.DateTimes = dts
        if not self.SectionIDs: self.SectionIDs = self._FT.getID()
        self.IDs = (self.SectionIDs if not ids else ids)
        self.DTRuler = (self.DTRuler if self.DTRuler else self._FT.getDateTime())
        self._OldArgs = {}
        self._OldArgs["清空缓存"], self.ClearCache = self.ClearCache, False
        self._initMode(**kwargs)
        self._isStarted = True

    # 结束批量模式
    def end(self):
        if not self._isStarted: return 0
        self.update(self._OldArgs)
        self._exit()
        self._isStarted = False
        return 0

# 因子表准备子进程
def _prepareRawData(args):
    nGroup = len(args['GroupInfo'])
    if "Sub2MainQueue" not in args:# 运行模式为串行
        with ProgressBar(max_value=nGroup) as ProgBar:
            for i in range(nGroup):
                if args["FT"]._QSArgs.OperationMode._Cache.createRawDataCache(args["RawDataFileNames"][i]): continue
                iFT, iFactorNames, iRawFactorNames, iDTs, iArgs = args['GroupInfo'][i]
                iPrepareIDs = args["PrepareIDs"][i]
                if iPrepareIDs is None: iPrepareIDs = args["FT"]._QSArgs.OperationMode.IDs
                iPID_PrepareIDs = args["PID_PrepareIDs"][i]
                if iPID_PrepareIDs is None: iPID_PrepareIDs = args["FT"]._QSArgs.OperationMode._PID_IDs
                iRawData = iFT.__QS_prepareRawData__(iRawFactorNames, iPrepareIDs, iDTs, iArgs)
                iFT.__QS_saveRawData__(iRawData, iRawFactorNames, args["FT"]._QSArgs.OperationMode._Cache, iPID_PrepareIDs, args["RawDataFileNames"][i])
                del iRawData
                gc.collect()
                ProgBar.update(i+1)
    else:# 运行模式为并行
        for i in range(nGroup):
            if args["FT"]._QSArgs.OperationMode._Cache.createRawDataCache(args["RawDataFileNames"][i]): continue
            iFT, iFactorNames, iRawFactorNames, iDTs, iArgs = args['GroupInfo'][i]
            iPrepareIDs = args["PrepareIDs"][i]
            if iPrepareIDs is None: iPrepareIDs = args["FT"]._QSArgs.OperationMode.IDs
            iPID_PrepareIDs = args["PID_PrepareIDs"][i]
            if iPID_PrepareIDs is None: iPID_PrepareIDs = args["FT"]._QSArgs.OperationMode._PID_IDs
            iRawData = iFT.__QS_prepareRawData__(iRawFactorNames, iPrepareIDs, iDTs, iArgs)
            iFT.__QS_saveRawData__(iRawData, iRawFactorNames, args["FT"]._QSArgs.OperationMode._Cache, iPID_PrepareIDs, args["RawDataFileNames"][i])
            del iRawData
            gc.collect()
            args['Sub2MainQueue'].put((args["PID"], 1, None))
    return 0
# 因子表运算子进程
def _calculate(args):
    FT = args["FT"]
    OperationMode = FT._QSArgs.OperationMode
    OperationMode._iPID = args["PID"]
    # 分配任务
    TDB, TableName, SpecificTarget = args["FactorDB"], args["TableName"], args["specific_target"]
    if SpecificTarget:
        TaskDispatched = OrderedDict()# {(id(FactorDB), TableName) : (FatorDB, [Factor], [FactorName])}
        for iFactorName in OperationMode.FactorNames:
            iDB, iTableName, iTargetFactorName = SpecificTarget.get(iFactorName, (None, None, None))
            if iDB is None: iDB = TDB
            if iTableName is None: iTableName = TableName
            if iTargetFactorName is None: iTargetFactorName = iFactorName
            iDBTable = (id(iDB), iTableName)
            if iDBTable in TaskDispatched:
                TaskDispatched[iDBTable][1].append(OperationMode._FactorDict[iFactorName])
                TaskDispatched[iDBTable][2].append(iTargetFactorName)
            else:
                TaskDispatched[iDBTable] = (iDB, [OperationMode._FactorDict[iFactorName]], [iTargetFactorName])
    else:
        TaskDispatched = {(id(TDB), TableName): (TDB, [OperationMode._FactorDict[iFactorName] for iFactorName in OperationMode.FactorNames], list(OperationMode.FactorNames))}
    # 执行任务
    nTask = len(OperationMode.FactorNames)
    nDT = len(OperationMode.DateTimes)
    TaskCount, BatchNum = 0, OperationMode.WriteBatchNum
    if OperationMode.SubProcessNum==0:# 运行模式为串行
        with ProgressBar(max_value=nTask) as ProgBar:
            for i, iTask in enumerate(TaskDispatched):
                iDB, iFactors, iTargetFactorNames = TaskDispatched[iTask]
                iTableName = iTask[1]
                if hasattr(iDB, "writeFactorData"):
                    for j, jFactor in enumerate(iFactors):
                        jData = jFactor._QS_getData(dts=OperationMode.DateTimes, pids=[args["PID"]])
                        if OperationMode._FactorPrepareIDs[jFactor.Name] is not None:
                            jData = jData.reindex(columns=OperationMode.IDs)
                        iDB.writeFactorData(jData, iTableName, iTargetFactorNames[j], if_exists=args["if_exists"], data_type=jFactor.getMetaData(key="DataType"), **args["kwargs"])
                        jData = None
                        TaskCount += 1
                        ProgBar.update(TaskCount)
                else:
                    iFactorNum = len(iFactors)
                    iBatchNum = (iFactorNum if BatchNum<=0 else BatchNum)
                    iDTLen= int(np.ceil(nDT / iBatchNum))
                    iDataTypes = {iTargetFactorNames[j]:jFactor.getMetaData(key="DataType") for j, jFactor in enumerate(iFactors)}
                    for j in range(iBatchNum):
                        jDTs = list(OperationMode.DateTimes[j*iDTLen:(j+1)*iDTLen])
                        if jDTs:
                            jData = {}
                            for k, kFactor in enumerate(iFactors):
                                ijkData = kFactor._QS_getData(dts=jDTs, pids=[args["PID"]])
                                if OperationMode._FactorPrepareIDs[kFactor.Name] is not None:
                                    ijkData = ijkData.reindex(columns=OperationMode.IDs)
                                jData[iTargetFactorNames[k]] = ijkData
                                if j==0:
                                    TaskCount += 0.5
                                    ProgBar.update(TaskCount)
                            jData = Panel(jData, items=iTargetFactorNames, major_axis=jDTs)
                            iDB.writeData(jData, iTableName, if_exists=args["if_exists"], data_type=iDataTypes, **args["kwargs"])
                            jData = None
                        TaskCount += 0.5 * iFactorNum / iBatchNum
                        ProgBar.update(TaskCount)
    else:
        for i, iTask in enumerate(TaskDispatched):
            iDB, iFactors, iTargetFactorNames = TaskDispatched[iTask]
            iTableName = iTask[1]
            if hasattr(iDB, "writeFactorData"):
                for j, jFactor in enumerate(iFactors):
                    if OperationMode._FactorPrepareIDs[jFactor.Name] is not None:
                        jData = jFactor._QS_getData(dts=OperationMode.DateTimes, pids=None)
                        jData = jData.reindex(columns=OperationMode._PID_IDs[args["PID"]])
                    else:
                        jData = jFactor._QS_getData(dts=OperationMode.DateTimes, pids=[args["PID"]])
                    iDB.writeFactorData(jData, iTableName, iTargetFactorNames[j], if_exists=args["if_exists"], data_type=jFactor.getMetaData(key="DataType"), **args["kwargs"])
                    jData = None
                    args["Sub2MainQueue"].put((args["PID"], 1, None))
            else:
                iFactorNum = len(iFactors)
                iBatchNum = (iFactorNum if BatchNum <= 0 else BatchNum)
                iDTLen= int(np.ceil(nDT / iBatchNum))
                iDataTypes = {iTargetFactorNames[j]:jFactor.getMetaData(key="DataType") for j, jFactor in enumerate(iFactors)}
                for j in range(iBatchNum):
                    jDTs = list(OperationMode.DateTimes[j*iDTLen:(j+1)*iDTLen])
                    if jDTs:
                        jData = {}
                        for k, kFactor in enumerate(iFactors):
                            ijkData = kFactor._QS_getData(dts=jDTs, pids=[args["PID"]])
                            if OperationMode._FactorPrepareIDs[kFactor.Name] is not None:
                                ijkData = ijkData.reindex(columns=FT._QSArgs.OperationMode.IDs)
                            jData[iTargetFactorNames[k]] = ijkData
                            if j==0: args["Sub2MainQueue"].put((args["PID"], 0.5, None))
                        jData = Panel(jData, items=iTargetFactorNames, major_axis=jDTs)
                        iDB.writeData(jData, iTableName, if_exists=args["if_exists"], data_type=iDataTypes, **args["kwargs"])
                        jData = None
                    args["Sub2MainQueue"].put((args["PID"], 0.5 * iFactorNum / iBatchNum, None))
    return 0

# 因子表, 接口类
# 因子表可看做一个独立的数据集或命名空间, 可看做 Panel(items=[因子], major_axis=[时间点], minor_axis=[ID])
# 因子表的数据有三个维度: 时间点, ID, 因子
# 时间点数据类型是 datetime.datetime, ID 和因子名称的数据类型是 str
# 不支持某个操作时, 方法产生错误
# 没有相关数据时, 方法返回 None
class FactorTable(__QS_Object__):
    """因子表"""
    class __QS_ArgClass__(__QS_Object__.__QS_ArgClass__):
        ErgodicMode = Instance(_ErgodicMode, arg_type="ArgObject", label="遍历模式", order=-3, eq_arg=False)
        OperationMode = Instance(_OperationMode, arg_type="ArgObject", label="批量模式", order=-4, eq_arg=False)
        def __QS_initArgs__(self, args={}):
            self.ErgodicMode = _ErgodicMode(owner=self._Owner, logger=self._QS_Logger)
            self.OperationMode = _OperationMode(owner=self._Owner, logger=self._QS_Logger)
    
    def __init__(self, name, fdb=None, sys_args={}, config_file=None, **kwargs):
        self._Name = name
        self._FactorDB = fdb# 因子表所属的因子库, None 表示自定义的因子表
        return super().__init__(sys_args=sys_args, config_file=config_file, **kwargs)
    
    @property
    def Name(self):
        return self._Name
    @property
    def FactorDB(self):
        return self._FactorDB
    def __enter__(self):
        if self._QSArgs.ErgodicMode._isStarted:
            self._QSArgs.ErgodicMode._OldArgs = {"自动缓冲": self._QSArgs.ErgodicMode.AutoMove}
            self._QSArgs.ErgodicMode.AutoMove = True
        elif self._QSArgs.OperationMode._isStarted:
            self._QS_Logger.debug(f"因子表 '{self._Name}' 开启批量运算模式")
        else:
            self._QS_Logger.warning(f"当前未开启任何运算模式!")
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        if self._QSArgs.ErgodicMode._isStarted:
            self._QSArgs.ErgodicMode.update(self._QSArgs.ErgodicMode._OldArgs)
        self.end()
        return (exc_type is None)
    # -------------------------------表的信息---------------------------------
    # 获取表的元数据
    def getMetaData(self, key=None, args={}):
        if key is None: return pd.Series()
        return None
    # -------------------------------维度信息-----------------------------------
    # 返回所有因子名
    @property
    def FactorNames(self):
        return []
    # 获取因子对象
    def getFactor(self, ifactor_name, args={}, new_name=None):
        iFactor = Factor(name=ifactor_name, ft=self, logger=self._QS_Logger)
        for iArgName in self._QSArgs.ArgNames:
            if iArgName not in ("遍历模式", "批量模式"):
                iTraitName, iTrait = self._QSArgs.getTrait(iArgName)
                iFactor._QSArgs._QS_Frozen = False
                iFactor._QSArgs.add_trait(iTraitName, iTrait)
                iFactor._QSArgs[iArgName] = args.get(iArgName, self._QSArgs[iArgName])
                iFactor._QSArgs._QS_Frozen = True
        if new_name is not None: iFactor.Name = new_name
        return iFactor
    
    # 查找因子对象, def_path: 以/分割的因子查找路径, 比如 年化收益率/0/1
    def searchFactor(self, factor_name=None, def_path=None, only_one=True, raise_error=True):
        if def_path is not None:
            def_path = def_path.split("/")
            if def_path[0] not in self.FactorNames:
                return None
            iFactor = self.getFactor(def_path[0])
            for iIdx in def_path[1:]:
                try:
                    iFactor = iFactor.Descriptors[int(iIdx)]
                except:
                    if raise_error:
                        raise __QS_Error__(f"查找不到因子: {def_path}")
                    return None
            if (factor_name is not None) and (iFactor.Name != factor_name):
                if raise_error:
                    if factor_name is not None:
                        raise __QS_Error__(f"查找不到因子({factor_name}): {def_path}")
                    else:
                        raise __QS_Error__(f"查找不到因子: {def_path}")
                return None
            else:
                return iFactor
        elif factor_name is not None:
            def _searchFactor(factors, factor_name):
                Factors = []
                for iFactor in factors:
                    if iFactor.Name == factor_name:
                        Factors.append(iFactor)
                    Factors += _searchFactor(iFactor.Descriptors, factor_name)
                return Factors

            Factors = []
            for iFactorName in self.FactorNames:
                iFactor = self.getFactor(iFactorName)
                if iFactorName == factor_name:
                    Factors.append(iFactor)
                Factors += _searchFactor(iFactor.Descriptors, factor_name)
            if only_one:
                if len(Factors) == 1:
                    return Factors[0]
                elif len(Factors) == 0:
                    if raise_error:
                        raise __QS_Error__(f"查找不到因子: {factor_name}")
                    else:
                        return None
                else:
                    if raise_error:
                        raise __QS_Error__(f"因子({factor_name}) 不止一个!")
                    else:
                        return None
            else:
                return Factors
        else:
            raise __QS_Error__("参数 def_path 和 factor_name 不能同时为 None!")

    # 获取因子的元数据
    def getFactorMetaData(self, factor_names, key=None, args={}):
        if key is None: return pd.DataFrame(index=factor_names, dtype=np.dtype("O"))
        else: return pd.Series([None]*len(factor_names), index=factor_names, dtype=np.dtype("O"))
    # 获取 ID 序列
    def getID(self, ifactor_name=None, idt=None, args={}):
        return []
    # 获取 ID 的 Mask, 返回: Series(True or False, index=[ID])
    def getIDMask(self, idt, ids=None, id_filter_str=None, args={}):
        if ids is None: ids = self.getID(idt=idt, args=args)
        if not id_filter_str: return pd.Series(True, index=ids)
        CompiledIDFilterStr, IDFilterFactors = testIDFilterStr(id_filter_str, self.FactorNames)
        if CompiledIDFilterStr is None: raise __QS_Error__("过滤条件字符串有误!")
        temp = self.readData(factor_names=IDFilterFactors, ids=ids, dts=[idt], args=args).loc[:, idt, :]
        return eval(CompiledIDFilterStr)
    # 获取过滤后的 ID
    def getFilteredID(self, idt, ids=None, id_filter_str=None, args={}):
        if not id_filter_str: return self.getID(idt=idt, args=args)
        if ids is None: ids = self.getID(idt=idt, args=args)
        CompiledIDFilterStr, IDFilterFactors = testIDFilterStr(id_filter_str, self.FactorNames)
        if CompiledIDFilterStr is None: raise __QS_Error__("过滤条件字符串有误!")
        temp = self.readData(factor_names=IDFilterFactors, ids=ids, dts=[idt], args=args).loc[:, idt, :]
        return eval("temp["+CompiledIDFilterStr+"].index.tolist()")
    # 获取时间点序列
    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None, args={}):
        return []
    # -------------------------------读取数据---------------------------------
    # 准备原始数据的接口
    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        return None
    # 计算数据的接口, 返回: Panel(item=[因子], major_axis=[时间点], minor_axis=[ID])
    def __QS_calcData__(self, raw_data, factor_names, ids, dts, args={}):
        return None
    # 读取数据, 返回: Panel(item=[因子], major_axis=[时间点], minor_axis=[ID])
    def readData(self, factor_names, ids, dts, args={}):
        if self._QSArgs.ErgodicMode._isStarted: return self._QSArgs.ErgodicMode.readData(factor_names=factor_names, ids=ids, dts=dts, args=args)
        if self._QSArgs.OperationMode._isStarted: return self._QSArgs.OperationMode.readData(factor_names=factor_names, ids=ids, dts=dts, args=args)
        return self.__QS_calcData__(raw_data=self.__QS_prepareRawData__(factor_names=factor_names, ids=ids, dts=dts, args=args), factor_names=factor_names, ids=ids, dts=dts, args=args)
    def __getitem__(self, key):
        if isinstance(key, str): return self.getFactor(key)
        elif isinstance(key, tuple): key += (slice(None),) * (3 - len(key))
        else: key = (key, slice(None), slice(None))
        if len(key)>3: raise IndexError("QuantStudio.FactorDataBase.FactorDB.FactorTable: Too many indexers")
        FactorNames, DTs, IDs = key
        if FactorNames==slice(None): FactorNames = self.FactorNames
        elif isinstance(FactorNames, str): FactorNames = [FactorNames]
        if DTs==slice(None): DTs = None
        elif isinstance(DTs, dt.datetime): DTs = [DTs]
        if IDs==slice(None): IDs = None
        elif isinstance(IDs, str): IDs = [IDs]
        Data = self.readData(FactorNames, IDs, DTs)
        return Data.loc[key]
    
    # ------------------------------------运算模式------------------------------------
    # 启动运算模式, dts: 遍历的时间点序列或者迭代器
    def start(self, dts, mode="遍历模式", ids=None, mode_args={}, **kwargs):
        if mode=="遍历模式":
            if self._QSArgs.OperationMode._isStarted:
                raise __QS_Error__("不能开启遍历模式, 当前处于批量模式下!")
            self._QSArgs.ErgodicMode.update(mode_args)
            self._QSArgs.ErgodicMode.start(dts=dts, **kwargs)
            self._QS_Logger.debug(f"因子表 '{self._Name}' 开启遍历运算模式")
        elif mode=="批量模式":
            if self._QSArgs.OperationMode._isStarted:
                raise __QS_Error__("不能开启批量模式, 当前处于遍历模式下!")
            self._QSArgs.OperationMode.update(mode_args)
            self._QSArgs.OperationMode.start(dts=dts, ids=ids, **kwargs)
            self._QS_Logger.debug(f"因子表 '{self._Name}' 开启批量运算模式")
        else:
            raise __QS_Error__(f"不支持 '{mode}', 可选: '遍历模式', '批量模式'")
        return self

    # 遍历模式下移动当前时点
    def move(self, idt, **kwargs):
        return self._QSArgs.ErgodicMode.move(idt, **kwargs)

    # 结束运算模式
    def end(self):
        if self._QSArgs.ErgodicMode._isStarted:
            self._QSArgs.ErgodicMode.end()
            self._QS_Logger.debug(f"因子表 '{self._Name}' 结束遍历运算模式")
        if self._QSArgs.OperationMode._isStarted:
            self._QSArgs.OperationMode.end()
            self._QS_Logger.debug(f"因子表 '{self._Name}' 结束批量运算模式")
        return 0

    # ------------------------------------批量模式------------------------------------
    # 获取因子表准备原始数据的分组信息, [(因子表对象, [因子名], [原始因子名], [时点], {参数})]
    def __QS_genGroupInfo__(self, factors, operation_mode):
        StartDT = dt.datetime.now()
        FactorNames, RawFactorNames = [], set()
        for iFactor in factors:
            FactorNames.append(iFactor.Name)
            RawFactorNames.add(iFactor._NameInFT)
            StartDT = min((StartDT, operation_mode._FactorStartDT[iFactor.Name]))
        EndDT = operation_mode.DateTimes[-1]
        StartInd, EndInd = operation_mode.DTRuler.index(StartDT), operation_mode.DTRuler.index(EndDT)
        return [(self, FactorNames, list(RawFactorNames), operation_mode.DTRuler[StartInd:EndInd+1], {})]
    def __QS_saveRawData__(self, raw_data, factor_names, cache: FactorCache, pid_ids, file_name, **kwargs):
        return cache.writeRawData(file_name, raw_data, target_fields=factor_names, additional_data=kwargs.get("additional_data", {}))

    # 计算因子数据并写入因子库
    # specific_target: {因子名: (目标因子库对象, 目标因子表名, 目标因子名)}
    # kwargs: 可选参数, 该参数同时传给因子库的 writeData 方法
    def write2FDB(self, factor_names, ids, dts, factor_db, table_name, if_exists="update", subprocess_num=cpu_count()-1, dt_ruler=None, section_ids=None, specific_target={}, **kwargs):
        Args = {"子进程数": subprocess_num, "时点标尺": dt_ruler, "截面ID": section_ids}
        return self._QSArgs.OperationMode.write2FDB(factor_names, ids, dts, factor_db, table_name, if_exists=if_exists, specific_target=specific_target, args=Args, **kwargs)
    
    def _repr_html_(self):
        HTML = f"<b>名称</b>: {html.escape(self.Name)}<br/>"
        HTML += f"<b>来源因子库</b>: {html.escape(self.FactorDB.Name) if self.FactorDB is not None else ''}<br/>"
        HTML += f"<b>因子列表</b>: {html.escape(str(self.FactorNames))}<br/>"
        MetaData = self.getMetaData()
        # MetaData = MetaData[~MetaData.index.str.contains("_QS")]
        MetaData = MetaData[~MetaData.index.astype(str).str.contains("_QS")]
        HTML += f"<b>元信息</b>: {dict2html(MetaData)}"
        return HTML + super()._repr_html_()
    
    def equals(self, other):
        if self is other: return True
        if not isinstance(other, FactorTable): return False
        if not (isinstance(other, type(self)) or isinstance(self, type(other))): return False
        if not self._FactorDB.equals(other._FactorDB): return False
        if not (self._Name != other._Name): return False
        if self._QSArgs != other._QSArgs: return False
        return True


# 自定义因子表
class CustomFT(FactorTable):
    """自定义因子表"""

    class __QS_ArgClass__(FactorTable.__QS_ArgClass__):
        MetaData = Dict({}, arg_type="Dict", label="元信息", order=-6, eq_arg=False)
    
    def __init__(self, name, sys_args={}, config_file=None, **kwargs):
        self._DateTimes = []# 数据源可提取的最长时点序列，[datetime.datetime]
        self._IDs = []# 数据源可提取的最长ID序列，['600000.SH']
        self._Factors = {}# 因子对象, {因子名: 因子对象}
        
        self._FactorDict = pd.DataFrame(columns=["FTID", "ArgIndex", "NameInFT", "DataType"], dtype=np.dtype("O"))# 数据源中因子的来源信息, index=[因子名]
        self._TableArgDict = {}# 数据源中的表和参数信息, {id(FT) : (FT, [args]), id(None) : ([Factor], [args])}
        
        self._IDFilterStr = None# ID 过滤条件字符串, "@收益率>0", 给定日期, 数据源的 getID 将返回过滤后的 ID 序列
        self._CompiledIDFilter = {}# 编译过的过滤条件字符串以及对应的因子列表, {条件字符串: (编译后的条件字符串,[因子])}
        self._isStarted = False# 数据源是否启动
        return super().__init__(name=name, fdb=None, sys_args=sys_args, config_file=config_file, **kwargs)
    @property
    def FactorNames(self):
        return sorted(self._Factors)
    def getMetaData(self, key=None, args={}):
        if key is None: return pd.Series(self._QSArgs.MetaData)
        return self._QSArgs.MetaData.get(key, None)
    def getFactorMetaData(self, factor_names=None, key=None, args={}):
        if factor_names is None: factor_names = self.FactorNames
        if key is not None: return pd.Series({iFactorName: self._Factors[iFactorName].getMetaData(key) for iFactorName in factor_names})
        else: return pd.DataFrame({iFactorName: self._Factors[iFactorName].getMetaData(key) for iFactorName in factor_names}).T
    def getFactor(self, ifactor_name, args={}, new_name=None):
        iFactor = self._Factors[ifactor_name]
        if new_name is not None: iFactor.Name = new_name
        return iFactor
    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None, args={}):
        if (start_dt is not None) or (end_dt is not None):
            DateTimes = np.array(self._DateTimes, dtype="O")
            if start_dt is not None: DateTimes = DateTimes[DateTimes>=start_dt]
            if end_dt is not None: DateTimes = DateTimes[DateTimes<=end_dt]
            return DateTimes.tolist()
        else:
            return self._DateTimes
    def getID(self, ifactor_name=None, idt=None, args={}):
        return self._IDs
    def getIDMask(self, idt, ids=None, id_filter_str=None, args={}):
        if ids is None: ids = self.getID(idt=idt, args=args)
        OldIDFilterStr = self.setIDFilter(id_filter_str)
        if self._IDFilterStr is None:
            self._IDFilterStr = OldIDFilterStr
            return pd.Series(True, index=ids)
        CompiledFilterStr, IDFilterFactors = self._CompiledIDFilter[self._IDFilterStr]
        temp = self.readData(factor_names=IDFilterFactors, ids=ids, dts=[idt], args=args).loc[:, idt, :]
        self._IDFilterStr = OldIDFilterStr
        return eval(CompiledFilterStr)
    def getFilteredID(self, idt, ids=None, id_filter_str=None, args={}):
        OldIDFilterStr = self.setIDFilter(id_filter_str)
        if ids is None: ids = self.getID(idt=idt, args=args)
        if self._IDFilterStr is None:
            self._IDFilterStr = OldIDFilterStr
            return ids
        CompiledFilterStr, IDFilterFactors = self._CompiledIDFilter[self._IDFilterStr]
        if CompiledFilterStr is None: raise __QS_Error__("过滤条件字符串有误!")
        temp = self.readData(factor_names=IDFilterFactors, ids=ids, dts=[idt], args=args).loc[:, idt, :]
        self._IDFilterStr = OldIDFilterStr
        return eval("temp["+CompiledFilterStr+"].index.tolist()")
    def __QS_calcData__(self, raw_data, factor_names, ids, dts, args={}):
        return Panel({iFactorName:self._Factors[iFactorName].readData(ids=ids, dts=dts, dt_ruler=self._DateTimes, section_ids=self._IDs) for iFactorName in factor_names}, items=factor_names, major_axis=dts, minor_axis=ids)
    def write2FDB(self, factor_names, ids, dts, factor_db, table_name, if_exists="update", subprocess_num=cpu_count()-1, dt_ruler=None, section_ids=None, specific_target={}, **kwargs):
        if dt_ruler is None: dt_ruler = self._DateTimes
        if not dt_ruler: dt_ruler = None
        if section_ids is None: section_ids = self._IDs
        if (not section_ids) or (section_ids==ids): section_ids = None
        return super().write2FDB(factor_names, ids, dts, factor_db, table_name, if_exists, subprocess_num, dt_ruler=dt_ruler, section_ids=section_ids, specific_target=specific_target, **kwargs)
    # ---------------新的接口------------------
    # 添加因子, factor_list: 因子对象列表
    def addFactors(self, factor_list=[], factor_table=None, factor_names=None, replace=True, args={}):
        if replace:
            FactorNames = {iFactor.Name for iFactor in factor_list}
            if factor_table is not None:
                FactorNames = set(FactorNames).union(factor_table.FactorNames if factor_names is None else factor_names)
            FactorNames = sorted(FactorNames.intersection(self.FactorNames))
            if FactorNames: self.deleteFactors(factor_names=FactorNames)
        for iFactor in factor_list:
            if iFactor.Name in self._Factors:
                raise __QS_Error__("因子: '%s' 有重名!" % iFactor.Name)
            self._Factors[iFactor.Name] = iFactor
        if factor_table is None: return 0
        if factor_names is None: factor_names = factor_table.FactorNames
        for iFactorName in factor_names:
            if iFactorName in self._Factors: raise __QS_Error__("因子: '%s' 有重名!" % iFactorName)
            iFactor = factor_table.getFactor(iFactorName, args=args)
            self._Factors[iFactor.Name] = iFactor
        return 0
    # 删除因子, factor_names = None 表示删除所有因子
    def deleteFactors(self, factor_names=None):
        if factor_names is None: factor_names = self.FactorNames
        for iFactorName in factor_names:
            if iFactorName not in self._Factors: continue
            self._Factors.pop(iFactorName, None)
        return 0
    # 重命名因子
    def renameFactor(self, factor_name, new_factor_name):
        if factor_name not in self._Factors: raise __QS_Error__("因子: '%s' 不存在!" % factor_name)
        if (new_factor_name!=factor_name) and (new_factor_name in self._Factors): raise __QS_Error__("因子: '%s' 有重名!" % new_factor_name)
        self._Factors[new_factor_name] = self._Factors.pop(factor_name)
        return 0
    # 设置时间点序列
    def setDateTime(self, dts):
        self._DateTimes = sorted(dts)
    # 设置 ID 序列
    def setID(self, ids):
        self._IDs = sorted(ids)
    # ID 过滤条件
    @property
    def IDFilterStr(self):
        return self._IDFilterStr
    # 设置 ID 过滤条件, id_filter_str, '@收益率$>0'
    def setIDFilter(self, id_filter_str):
        OldIDFilterStr = self._IDFilterStr
        if not id_filter_str:
            self._IDFilterStr = None
            return OldIDFilterStr
        elif not isinstance(id_filter_str, str): raise __QS_Error__("条件字符串必须为字符串或者为 None!")
        CompiledIDFilter = self._CompiledIDFilter.get(id_filter_str, None)
        if CompiledIDFilter is not None:# 该条件已经编译过
            self._IDFilterStr = id_filter_str
            return OldIDFilterStr
        CompiledIDFilterStr, IDFilterFactors = testIDFilterStr(id_filter_str, self.FactorNames)
        if CompiledIDFilterStr is None: raise __QS_Error__(f"条件字符串有误: {id_filter_str}")
        self._IDFilterStr = id_filter_str
        self._CompiledIDFilter[id_filter_str] = (CompiledIDFilterStr, IDFilterFactors)
        return OldIDFilterStr
    def start(self, dts, mode="遍历模式", ids=None, mode_args={}, **kwargs):
        super().start(dts=dts, mode=mode, ids=ids, mode_args=mode_args, **kwargs)
        if mode=="遍历模式":
            ModeArgs = self._QSArgs.ErgodicMode
        elif mode=="批量模式":
            ModeArgs = self._QSArgs.OperationMode
        for iFactor in self._Factors.values(): iFactor.start(dts=dts, mode=mode, ids=ids, mode_args=ModeArgs, **kwargs)
        return self
    def end(self):
        super().end()
        for iFactor in self._Factors.values(): iFactor.end()
        return 0
    
    def equals(self, other):
        if self is other: return True
        if not super().equals(other): return False
        if self._DateTimes != other._DateTimes: return False
        if self._IDs != other._IDs: return False
        if self._Factors != other._Factors: return False
        return True


# ---------- 内置的因子运算----------
# 将运算结果转换成真正的可以存储的因子
def Factorize(factor_object, factor_name, args={}, **kwargs):
    factor_object.Name = factor_name
    for iArg, iVal in args.items(): factor_object._QSArgs[iArg] = iVal
    if "logger" in kwargs: factor_object._QS_Logger = kwargs["logger"]
    return factor_object

def _toExpr(obj):
    if isinstance(obj, sympy.Expr):
        return obj
    elif isinstance(obj, sympy.logic.boolalg.Boolean):
        return sympy.Function("I")(obj)
    elif isinstance(obj, str):
        return sympy.Symbol(f"'{obj}'")
    else:
        # raise __QS_Error__(f"{obj} 转 sympy.Expr 失败, 不支持的 sympy 类型: {type(obj)}")
        return obj

def _toBoolean(obj):
    if isinstance(obj, sympy.Symbol):
        return sympy.Eq(obj, 1)
    elif isinstance(obj, sympy.logic.boolalg.Boolean):
        return obj
    elif isinstance(obj, sympy.Expr):
        return sympy.Eq(obj, 1)
    else:
        # raise __QS_Error__(f"{obj} 转 sympy.logic.boolalg.Boolean 失败, 不支持的 sympy 类型: {type(obj)}")
        return obj

def _UnitaryOperator(f, idt, iid, x, args):
    Fun = args.get("Fun", None)
    if Fun is not None: Data = Fun(f, idt, iid, x, args["Arg"])
    else: Data = x[0]
    OperatorType = args.get("OperatorType", "neg")
    if OperatorType=="neg": return -Data
    elif OperatorType=="abs": return np.abs(Data)
    elif OperatorType=="not": return (~Data)
    else: raise __QS_Error__("尚不支持的单因子运算符: %s" % OperatorType)

def _BinaryOperator(f, idt, iid, x, args):
    Fun1 = args.get("Fun1", None)
    if Fun1 is not None:
        Data1 = Fun1(f, idt, iid, x[:args["SepInd"]], args["Arg1"])
    else:
        Data1 = args.get("Data1", None)
        if Data1 is None: Data1 = x[0]
    Fun2 = args.get("Fun2",None)
    if Fun2 is not None:
        Data2 = Fun2(f, idt, iid, x[args["SepInd"]:], args["Arg2"])
    else:
        Data2 = args.get("Data2", None)
        if Data2 is None: Data2 = x[args["SepInd"]]
    OperatorType = args.get("OperatorType", "add")
    if OperatorType=="add": return Data1 + Data2
    elif OperatorType=="sub": return Data1 - Data2
    elif OperatorType=="mul": return Data1 * Data2
    elif OperatorType=="div":
        if np.isscalar(Data2): return (Data1 / Data2 if Data2!=0 else np.empty(Data1.shape)+np.nan)
        Data2[Data2==0] = np.nan
        return Data1/Data2
    elif OperatorType=="floordiv": return Data1 // Data2
    elif OperatorType=="mod": return Data1 % Data2
    elif OperatorType=="pow":
        if np.isscalar(Data2):
            if Data2<0: Data1[Data1==0] = np.nan
            return Data1 ** Data2
        if np.isscalar(Data1):
            if Data1==0: Data2[Data2<0] = np.nan
            return Data1 ** Data2
        Data1[(Data1==0) & (Data2<0)] = np.nan
        return Data1 ** Data2
    elif OperatorType=="and": return (Data1 & Data2)
    elif OperatorType=="or": return (Data1 | Data2)
    elif OperatorType=="xor": return (Data1 ^ Data2)
    elif OperatorType=="<": return (Data1 < Data2)
    elif OperatorType=="<=": return (Data1 <= Data2)
    elif OperatorType==">": return (Data1 > Data2)
    elif OperatorType==">=": return (Data1 >= Data2)
    elif OperatorType=="==": return (Data1 == Data2)
    elif OperatorType=="!=": return (Data1 != Data2)
    else: raise __QS_Error__("尚不支持的多因子运算符: %s" % OperatorType)

# 因子
# 因子可看做一个 DataFrame(index=[时间点], columns=[ID])
# 时间点数据类型是 datetime.datetime, ID 的数据类型是 str
# 不支持某个操作时, 方法产生错误
# 没有相关数据时, 方法返回 None
class Factor(__QS_Object__):
    """因子"""
    def __init__(self, name, ft, sys_args={}, config_file=None, **kwargs):
        self._FactorTable = ft# 因子所属的因子表, None 表示衍生因子
        self._NameInFT = name# 因子在所属的因子表中的名字
        self.Name = name# 因子对外显示的名称
        # 遍历模式下的对象
        self._isStarted = False# 是否启动了遍历模式
        self._CacheData = None# 遍历模式下缓存的数据
        self._FactorCache = None# 遍历模式下缓存对象, 如果为 None 表示使用内存缓存(_CacheData)
        # 批量模式下的对象
        self._OperationMode = None# 批量模式对象
        self._RawDataFile = ""# 原始数据存放地址
        self._isCacheDataOK = False# 是否准备好了缓存数据
        return super().__init__(sys_args=sys_args, config_file=config_file, **kwargs)
    @property
    def FactorTable(self):
        return self._FactorTable
    @property
    def Descriptors(self):
        return []
    # 获取因子的元数据
    def getMetaData(self, key=None, args={}):
        Args = self.Args
        Args.update(args)
        return self._FactorTable.getFactorMetaData(factor_names=[self._NameInFT], key=key, args=Args).loc[self._NameInFT]
    # 获取 ID 序列
    def getID(self, idt=None):
        if (self._OperationMode is not None) and (self._OperationMode._isStarted): return self._OperationMode.IDs
        if self._FactorTable is not None: return self._FactorTable.getID(ifactor_name=self._NameInFT, idt=idt, args=self.Args)
        return []
    # 获取时间点序列
    def getDateTime(self, iid=None, start_dt=None, end_dt=None):
        if (self._OperationMode is not None) and (self._OperationMode._isStarted): return self._OperationMode.DateTimes
        if self._FactorTable is not None: return self._FactorTable.getDateTime(ifactor_name=self._NameInFT, iid=iid, start_dt=start_dt, end_dt=end_dt, args=self.Args)
        return []
    # --------------------------------数据读取---------------------------------
    # 读取数据, 返回: Panel(item=[因子], major_axis=[时间点], minor_axis=[ID])
    def readData(self, ids, dts, **kwargs):
        if not self._isStarted: return self._FactorTable.readData(factor_names=[self._NameInFT], ids=ids, dts=dts, args=self.Args).loc[self._NameInFT]
        # 启动了遍历模式
        if self._FactorCache is not None:
            CacheFileName = self.Name + str(id(self))
            CacheData = self._FactorCache.readFactorData(CacheFileName, wait=False)
            if CacheData is None:
                CacheData = self._FactorTable.readData(factor_names=[self._NameInFT], ids=ids, dts=dts, args=self.Args).loc[self._NameInFT]
                self._FactorCache.writeFactorData(CacheFileName, CacheData, pid="0-0")
                return CacheData
        elif self._CacheData is None:
            if self._CacheData is None:
                CacheData = self._FactorTable.readData(factor_names=[self._NameInFT], ids=ids, dts=dts, args=self.Args).loc[self._NameInFT]
                self._CacheData = CacheData.copy()
                return CacheData
            else:
                CacheData = self._CacheData
        NewDTs = sorted(set(dts).difference(CacheData.index))
        if NewDTs:
            NewCacheData = self._FactorTable.readData(factor_names=[self._NameInFT], ids=CacheData.columns.tolist(), dts=NewDTs, args=self.Args).loc[self._NameInFT]
            CacheData = CacheData.append(NewCacheData).reindex(index=dts)
        NewIDs = sorted(set(ids).difference(CacheData.columns))
        if NewIDs:
            NewCacheData = self._FactorTable.readData(factor_names=[self._NameInFT], ids=NewIDs, dts=self._CacheData.index.tolist(), args=self.Args).loc[self._NameInFT]
            CacheData = pd.merge(CacheData, NewCacheData, left_index=True, right_index=True)
        if self._FactorCache is not None:
            self._FactorCache.writeFactorData(CacheFileName, CacheData, pid="0-0")
        else:
            self._CacheData = CacheData
        return CacheData.reindex(index=dts, columns=ids)
    def __getitem__(self, key):
        if isinstance(key, tuple): key += (slice(None),) * (2 - len(key))
        else: key = (key, slice(None))
        if len(key)>2: raise IndexError("QuantStudio.FactorDataBase.FactorDB.Factor: Too many indexers")
        DTs, IDs = key
        if DTs==slice(None): DTs = None
        elif isinstance(DTs, dt.datetime): DTs = [DTs]
        if IDs==slice(None): IDs = None
        elif isinstance(IDs, str): IDs = [IDs]
        Data = self.readData(IDs, DTs)
        return Data.loc[key]
    # ------------------------------------批量模式------------------------------------
    # 获取数据的开始时点, start_dt:新起始时点, dt_dict: 当前所有因子的时点信息: {因子名 : 开始时点}, id_dict: 当前所有因子的准备原始数据的截面 ID 信息: {因子名 : ID 序列}
    def _QS_initOperation(self, start_dt, dt_dict, prepare_ids, id_dict):
        OldStartDT = dt_dict.get(self.Name, start_dt)
        dt_dict[self.Name] = (start_dt if start_dt<OldStartDT else OldStartDT)
        PrepareIDs = id_dict.setdefault(self.Name, prepare_ids)
        if prepare_ids != PrepareIDs:
            raise __QS_Error__("因子 %s 指定了不同的截面!" % self.Name)
    # 准备缓存数据
    def __QS_prepareCacheData__(self, ids=None):
        StartDT = self._OperationMode._FactorStartDT[self.Name]
        EndDT = self._OperationMode.DateTimes[-1]
        StartInd, EndInd = self._OperationMode.DTRuler.index(StartDT), self._OperationMode.DTRuler.index(EndDT)
        DTs = self._OperationMode.DTRuler[StartInd:EndInd+1]
        RawData, PrepareIDs = self._OperationMode._Cache.readRawData(self._RawDataFile, self._OperationMode._iPID, self._NameInFT)
        if PrepareIDs is None:
            PrepareIDs = self._OperationMode._FactorPrepareIDs[self.Name]
            if PrepareIDs is None: PrepareIDs = self._OperationMode._PID_IDs[self._OperationMode._iPID]
        if RawData is not None:
            StdData = self._FactorTable.__QS_calcData__(RawData, factor_names=[self._NameInFT], ids=PrepareIDs, dts=DTs, args=self.Args).iloc[0]
        else:
            StdData = self._FactorTable.readData(factor_names=[self._NameInFT], ids=PrepareIDs, dts=DTs, args=self.Args).iloc[0]
        self._OperationMode._Cache.writeFactorData(self.Name+str(self._OperationMode._FactorID[self.Name]), StdData, pid=self._OperationMode._iPID)
        self._isCacheDataOK = True
        return StdData
    # 获取因子数据, pid=None表示取所有进程的数据
    def _QS_getData(self, dts, pids=None, **kwargs):
        if not self._isCacheDataOK:# 若没有准备好缓存数据, 准备缓存数据
            self.__QS_prepareCacheData__()
        StdData = self._OperationMode._Cache.readFactorData(self.Name+str(self._OperationMode._FactorID[self.Name]), pids=pids)
        if pids is not None:
            StdData = StdData.reindex(index=list(dts))
        elif self._OperationMode._FactorPrepareIDs[self.Name] is None:
            StdData = StdData.reindex(index=list(dts), columns=self._OperationMode.IDs)
        else:
            StdData = StdData.reindex(index=list(dts), columns=self._OperationMode._FactorPrepareIDs[self.Name])
        gc.collect()
        return StdData
    def _exit(self):
        self._OperationMode = None# 批量模式对象
        self._RawDataFile = ""# 原始数据存放地址
        self._isCacheDataOK = False# 是否准备好了缓存数据
    # ------------------------------------遍历模式------------------------------------
    # 启动遍历模式, dts: 遍历的时间点序列或者迭代器
    def start(self, dts, mode="遍历模式", ids=None, mode_args={}, **kwargs):
        if mode=="遍历模式":
            self._isStarted = True
            if mode_args.FactorCache is not None:
                self._FactorCache = mode_args.FactorCache
        return 0
    # 结束遍历模式
    def end(self):
        self._CacheData = None
        self._FactorCache = None
        self._isStarted = False
        return 0
    # -----------------------------重载运算符-------------------------------------
    def _genUnitaryOperatorInfo(self):
        if (self.Name==""):# 因子为中间运算因子
            Args = {"Fun":self._QSArgs.Operator, "Arg":self._QSArgs.ModelArgs}
            Exprs = self._QSArgs.Expression
            return (self.Descriptors, Args, Exprs)
        else:# 因子为正常因子
            Exprs = sympy.Symbol("_d1")
            return ([self], {}, Exprs)
    def _genBinaryOperatorInfo(self, other):
        if isinstance(other, Factor):# 两个因子运算
            if (self.Name=="") and (other.Name==""):# 两个因子因子名为空, 说明都是中间运算因子
                Args = {"Fun1":self._QSArgs.Operator, "Fun2":other._QSArgs.Operator, "SepInd":len(self.Descriptors), "Arg1":self._QSArgs.ModelArgs, "Arg2":other._QSArgs.ModelArgs}
                Exprs = [self._QSArgs.Expression, other._QSArgs.Expression]
                nDescriptor = len(self.Descriptors)
                for i in range(len(other.Descriptors), 0, -1):
                    Exprs[1] = Exprs[1].subs(sympy.Symbol(f"_d{i}"), sympy.Symbol(f"_d{nDescriptor+i}"))
                return (self.Descriptors+other.Descriptors, Args, Exprs)
            elif (self.Name==""):# 第一个因子为中间运算因子
                Args = {"Fun1":self._QSArgs.Operator, "SepInd":len(self.Descriptors), "Arg1":self._QSArgs.ModelArgs}
                Exprs = [self._QSArgs.Expression, sympy.Symbol(f"_d{len(self.Descriptors)+1}")]
                return (self.Descriptors+[other], Args, Exprs)
            elif (other.Name==""):# 第二个因子为中间运算因子
                Args = {"Fun2":other._QSArgs.Operator, "SepInd":1, "Arg2":other._QSArgs.ModelArgs}
                Exprs = [sympy.Symbol(f"_d1"), other._QSArgs.Expression]
                for i in range(len(other.Descriptors), 0, -1):
                    Exprs[1] = Exprs[1].subs(sympy.Symbol(f"_d{i}"), sympy.Symbol(f"_d{i+1}"))
                return ([self]+other.Descriptors, Args, Exprs)
            else:# 两个因子均为正常因子
                Args = {"SepInd":1}
                Exprs = [sympy.Symbol(f"_d1"), sympy.Symbol(f"_d2")]
                return ([self, other], Args, Exprs)
        elif (self.Name==""):# 中间运算因子+标量数据
            Args = {"Fun1":self._QSArgs.Operator, "SepInd":len(self.Descriptors), "Data2":other, "Arg1":self._QSArgs.ModelArgs}
            Exprs = [self._QSArgs.Expression, other]
            return (self.Descriptors, Args, Exprs)
        else:# 正常因子+标量数据
            Args = {"SepInd":1, "Data2":other}
            Exprs = [sympy.Symbol(f"_d1"), other]
            return ([self], Args, Exprs)
    def _genRBinaryOperatorInfo(self, other):
        if (self.Name==""):# 标量数据+中间运算因子
            Args = {"Fun2":self._QSArgs.Operator, "SepInd":0, "Data1":other, "Arg2":self._QSArgs.ModelArgs}
            Exprs = [other, self._QSArgs.Expression]
            return (self.Descriptors, Args, Exprs)
        else:# 标量数据+正常因子
            Args = {"SepInd":0, "Data1":other}
            Exprs = [other, sympy.Symbol("_d1")]
            return ([self], Args, Exprs)
    def __add__(self, other):
        from QuantStudio.FactorDataBase.FactorOperation import PointOperation
        Descriptors, Args, Exprs = self._genBinaryOperatorInfo(other)
        Args["OperatorType"] = "add"
        return PointOperation("", Descriptors, {"算子":_BinaryOperator, "参数":Args, "运算时点":"多时点", "运算ID":"多ID", "表达式": _toExpr(Exprs[0]) + _toExpr(Exprs[1])}, logger=self._QS_Logger)
    def __radd__(self, other):
        from QuantStudio.FactorDataBase.FactorOperation import PointOperation
        Descriptors, Args, Exprs = self._genRBinaryOperatorInfo(other)
        Args["OperatorType"] = "add"
        return PointOperation("", Descriptors, {"算子":_BinaryOperator, "参数":Args, "运算时点":"多时点", "运算ID":"多ID", "表达式": _toExpr(Exprs[0]) + _toExpr(Exprs[1])}, logger=self._QS_Logger)
    def __sub__(self, other):
        from QuantStudio.FactorDataBase.FactorOperation import PointOperation
        Descriptors, Args, Exprs = self._genBinaryOperatorInfo(other)
        Args["OperatorType"] = "sub"
        return PointOperation("", Descriptors, {"算子":_BinaryOperator, "参数":Args, "运算时点":"多时点", "运算ID":"多ID", "表达式": _toExpr(Exprs[0]) - _toExpr(Exprs[1])}, logger=self._QS_Logger)
    def __rsub__(self, other):
        from QuantStudio.FactorDataBase.FactorOperation import PointOperation
        Descriptors, Args, Exprs = self._genRBinaryOperatorInfo(other)
        Args["OperatorType"] = "sub"
        return PointOperation("", Descriptors, {"算子":_BinaryOperator, "参数":Args, "运算时点":"多时点", "运算ID":"多ID", "表达式": _toExpr(Exprs[0]) - _toExpr(Exprs[1])}, logger=self._QS_Logger)
    def __mul__(self, other):
        from QuantStudio.FactorDataBase.FactorOperation import PointOperation
        Descriptors,Args, Exprs = self._genBinaryOperatorInfo(other)
        Args["OperatorType"] = "mul"
        return PointOperation("", Descriptors, {"算子":_BinaryOperator, "参数":Args, "运算时点":"多时点", "运算ID":"多ID", "表达式": _toExpr(Exprs[0]) * _toExpr(Exprs[1])}, logger=self._QS_Logger)
    def __rmul__(self, other):
        from QuantStudio.FactorDataBase.FactorOperation import PointOperation
        Descriptors, Args, Exprs = self._genRBinaryOperatorInfo(other)
        Args["OperatorType"] = "mul"
        return PointOperation("", Descriptors, {"算子":_BinaryOperator, "参数":Args, "运算时点":"多时点", "运算ID":"多ID", "表达式": _toExpr(Exprs[0]) * _toExpr(Exprs[1])}, logger=self._QS_Logger)
    def __pow__(self, other):
        from QuantStudio.FactorDataBase.FactorOperation import PointOperation
        Descriptors, Args, Exprs = self._genBinaryOperatorInfo(other)
        Args["OperatorType"] = "pow"
        return PointOperation("", Descriptors, {"算子":_BinaryOperator, "参数":Args, "运算时点":"多时点", "运算ID":"多ID", "表达式": _toExpr(Exprs[0]) ** _toExpr(Exprs[1])}, logger=self._QS_Logger)
    def __rpow__(self, other):
        from QuantStudio.FactorDataBase.FactorOperation import PointOperation
        Descriptors, Args, Exprs = self._genRBinaryOperatorInfo(other)
        Args["OperatorType"] = "pow"
        return PointOperation("", Descriptors, {"算子":_BinaryOperator, "参数":Args, "运算时点":"多时点", "运算ID":"多ID", "表达式": _toExpr(Exprs[0]) ** _toExpr(Exprs[1])}, logger=self._QS_Logger)
    def __truediv__(self, other):
        from QuantStudio.FactorDataBase.FactorOperation import PointOperation
        Descriptors, Args, Exprs = self._genBinaryOperatorInfo(other)
        Args["OperatorType"] = "div"
        return PointOperation("", Descriptors, {"算子":_BinaryOperator, "参数":Args, "运算时点":"多时点", "运算ID":"多ID", "表达式": _toExpr(Exprs[0]) / _toExpr(Exprs[1])}, logger=self._QS_Logger)
    def __rtruediv__(self, other):
        from QuantStudio.FactorDataBase.FactorOperation import PointOperation
        Descriptors, Args, Exprs = self._genRBinaryOperatorInfo(other)
        Args["OperatorType"] = "div"
        return PointOperation("", Descriptors, {"算子":_BinaryOperator, "参数":Args, "运算时点":"多时点", "运算ID":"多ID", "表达式": _toExpr(Exprs[0]) / _toExpr(Exprs[1])}, logger=self._QS_Logger)
    def __floordiv__(self, other):
        from QuantStudio.FactorDataBase.FactorOperation import PointOperation
        Descriptors, Args, Exprs = self._genBinaryOperatorInfo(other)
        Args["OperatorType"] = "floordiv"
        return PointOperation("", Descriptors, {"算子":_BinaryOperator, "参数":Args, "运算时点":"多时点", "运算ID":"多ID", "表达式": _toExpr(Exprs[0]) // _toExpr(Exprs[1])}, logger=self._QS_Logger)
    def __rfloordiv__(self, other):
        from QuantStudio.FactorDataBase.FactorOperation import PointOperation
        Descriptors, Args, Exprs = self._genRBinaryOperatorInfo(other)
        Args["OperatorType"] = "floordiv"
        return PointOperation("", Descriptors, {"算子":_BinaryOperator, "参数":Args, "运算时点":"多时点", "运算ID":"多ID", "表达式": _toExpr(Exprs[0]) // _toExpr(Exprs[1])}, logger=self._QS_Logger)
    def __mod__(self, other):
        from QuantStudio.FactorDataBase.FactorOperation import PointOperation
        Descriptors, Args, Exprs = self._genBinaryOperatorInfo(other)
        Args["OperatorType"] = "mod"
        return PointOperation("", Descriptors, {"算子":_BinaryOperator, "参数":Args, "运算时点":"多时点", "运算ID":"多ID", "表达式": _toExpr(Exprs[0]) % _toExpr(Exprs[1])}, logger=self._QS_Logger)
    def __rmod__(self, other):
        from QuantStudio.FactorDataBase.FactorOperation import PointOperation
        Descriptors, Args, Exprs = self._genRBinaryOperatorInfo(other)
        Args["OperatorType"] = "mod"
        return PointOperation("", Descriptors, {"算子":_BinaryOperator, "参数":Args, "运算时点":"多时点", "运算ID":"多ID", "表达式": _toExpr(Exprs[0]) % _toExpr(Exprs[1])}, logger=self._QS_Logger)
    def __and__(self, other):
        from QuantStudio.FactorDataBase.FactorOperation import PointOperation
        Descriptors, Args, Exprs = self._genBinaryOperatorInfo(other)
        Args["OperatorType"] = "and"
        return PointOperation("", Descriptors, {"算子":_BinaryOperator, "参数":Args, "运算时点":"多时点", "运算ID":"多ID", "表达式": _toBoolean(Exprs[0]) & _toBoolean(Exprs[1])}, logger=self._QS_Logger)
    def __rand__(self, other):
        from QuantStudio.FactorDataBase.FactorOperation import PointOperation
        Descriptors, Args, Exprs = self._genRBinaryOperatorInfo(other)
        Args["OperatorType"] = "and"
        return PointOperation("", Descriptors, {"算子":_BinaryOperator, "参数":Args, "运算时点":"多时点", "运算ID":"多ID", "表达式": _toBoolean(Exprs[0]) & _toBoolean(Exprs[1])}, logger=self._QS_Logger)
    def __or__(self, other):
        from QuantStudio.FactorDataBase.FactorOperation import PointOperation
        Descriptors, Args, Exprs = self._genBinaryOperatorInfo(other)
        Args["OperatorType"] = "or"
        return PointOperation("", Descriptors, {"算子":_BinaryOperator, "参数":Args, "运算时点":"多时点", "运算ID":"多ID", "表达式": _toBoolean(Exprs[0]) | _toBoolean(Exprs[1])}, logger=self._QS_Logger)
    def __ror__(self, other):
        from QuantStudio.FactorDataBase.FactorOperation import PointOperation
        Descriptors, Args, Exprs = self._genRBinaryOperatorInfo(other)
        Args["OperatorType"] = "or"
        return PointOperation("", Descriptors, {"算子":_BinaryOperator, "参数":Args, "运算时点":"多时点", "运算ID":"多ID", "表达式": _toBoolean(Exprs[0]) | _toBoolean(Exprs[1])}, logger=self._QS_Logger)
    def __xor__(self, other):
        from QuantStudio.FactorDataBase.FactorOperation import PointOperation
        Descriptors, Args, Exprs = self._genBinaryOperatorInfo(other)
        Args["OperatorType"] = "xor"
        return PointOperation("", Descriptors, {"算子":_BinaryOperator, "参数":Args, "运算时点":"多时点", "运算ID":"多ID", "表达式": _toBoolean(Exprs[0]) ^ _toBoolean(Exprs[1])}, logger=self._QS_Logger)
    def __rxor__(self, other):
        from QuantStudio.FactorDataBase.FactorOperation import PointOperation
        Descriptors, Args, Exprs = self._genRBinaryOperatorInfo(other)
        Args["OperatorType"] = "xor"
        return PointOperation("", Descriptors, {"算子":_BinaryOperator, "参数":Args, "运算时点":"多时点", "运算ID":"多ID", "表达式": _toBoolean(Exprs[0]) ^ _toBoolean(Exprs[1])}, logger=self._QS_Logger)
    def __lt__(self, other):
        from QuantStudio.FactorDataBase.FactorOperation import PointOperation
        Descriptors, Args, Exprs = self._genBinaryOperatorInfo(other)
        Args["OperatorType"] = "<"
        return PointOperation("", Descriptors, {"算子":_BinaryOperator, "参数":Args, "运算时点":"多时点", "运算ID":"多ID", "表达式": _toExpr(Exprs[0]) < _toExpr(Exprs[1])}, logger=self._QS_Logger)
    def __le__(self, other):
        from QuantStudio.FactorDataBase.FactorOperation import PointOperation
        Descriptors, Args, Exprs = self._genBinaryOperatorInfo(other)
        Args["OperatorType"] = "<="
        return PointOperation("", Descriptors, {"算子":_BinaryOperator, "参数":Args, "运算时点":"多时点", "运算ID":"多ID", "表达式": _toExpr(Exprs[0]) < _toExpr(Exprs[1])}, logger=self._QS_Logger)
    def __eq__(self, other):
        from QuantStudio.FactorDataBase.FactorOperation import PointOperation
        Descriptors, Args, Exprs = self._genBinaryOperatorInfo(other)
        Args["OperatorType"] = "=="
        return PointOperation("", Descriptors, {"算子":_BinaryOperator, "参数":Args, "运算时点":"多时点", "运算ID":"多ID", "表达式": sympy.Eq(_toExpr(Exprs[0]), _toExpr(Exprs[1]))}, logger=self._QS_Logger)
    def __ne__(self, other):
        from QuantStudio.FactorDataBase.FactorOperation import PointOperation
        Descriptors, Args, Exprs = self._genBinaryOperatorInfo(other)
        Args["OperatorType"] = "!="
        return PointOperation("", Descriptors, {"算子":_BinaryOperator, "参数":Args, "运算时点":"多时点", "运算ID":"多ID", "表达式": sympy.Unequality(_toExpr(Exprs[0]), _toExpr(Exprs[1]))}, logger=self._QS_Logger)
    def __gt__(self, other):
        from QuantStudio.FactorDataBase.FactorOperation import PointOperation
        Descriptors, Args, Exprs = self._genBinaryOperatorInfo(other)
        Args["OperatorType"] = ">"
        return PointOperation("", Descriptors, {"算子":_BinaryOperator, "参数":Args, "运算时点":"多时点", "运算ID":"多ID", "表达式": _toExpr(Exprs[0]) > _toExpr(Exprs[1])}, logger=self._QS_Logger)
    def __ge__(self, other):
        from QuantStudio.FactorDataBase.FactorOperation import PointOperation
        Descriptors, Args, Exprs = self._genBinaryOperatorInfo(other)
        Args["OperatorType"] = ">="
        return PointOperation("", Descriptors, {"算子":_BinaryOperator, "参数":Args, "运算时点":"多时点", "运算ID":"多ID", "表达式": _toExpr(Exprs[0]) >= _toExpr(Exprs[1])}, logger=self._QS_Logger)
    def __neg__(self):
        from QuantStudio.FactorDataBase.FactorOperation import PointOperation
        Descriptors, Args, Exprs = self._genUnitaryOperatorInfo()
        Args["OperatorType"] = "neg"
        return PointOperation("", Descriptors, {"算子":_UnitaryOperator, "参数":Args, "运算时点":"多时点", "运算ID":"多ID", "表达式": - _toExpr(Exprs)}, logger=self._QS_Logger)
    def __pos__(self):
        return self
    def __abs__(self):
        from QuantStudio.FactorDataBase.FactorOperation import PointOperation
        Descriptors, Args, Exprs = self._genUnitaryOperatorInfo()
        Args["OperatorType"] = "abs"
        return PointOperation("", Descriptors, {"算子":_UnitaryOperator, "参数":Args, "运算时点":"多时点", "运算ID":"多ID", "表达式": abs(_toExpr(Exprs))}, logger=self._QS_Logger)
    def __invert__(self):
        from QuantStudio.FactorDataBase.FactorOperation import PointOperation
        Descriptors, Args, Exprs = self._genUnitaryOperatorInfo()
        Args["OperatorType"] = "not"
        return PointOperation("", Descriptors, {"算子":_UnitaryOperator, "参数":Args, "运算时点":"多时点", "运算ID":"多ID", "表达式": ~ _toBoolean(Exprs)}, logger=self._QS_Logger)

    def expression(self, penetrated=False):
        return sympy.Symbol(self.Name)

    def _repr_html_(self):
        HTML = f"<b>名称</b>: {html.escape(self.Name)}<br/>"
        HTML += f"<b>来源因子表</b>: {html.escape(self.FactorTable.Name) if self.FactorTable is not None else ''}<br/>"
        HTML += f"<b>原始名称</b>: {html.escape(self._NameInFT) if self.FactorTable is not None else ''}<br/>"
        HTML += f"<b>描述子列表</b>: {html.escape(str([iFactor.Name for iFactor in self.Descriptors]))}<br/>"
        Expr = self.expression(penetrated=False)
        Mappings = {iSymbol: iSymbol.name.replace("_", r"\_") for iSymbol in Expr.free_symbols}
        HTML += f"<b>表达式</b>: {html.escape(Math(sympy.latex(Expr, symbol_names=Mappings, mul_symbol='times'))._repr_latex_())}<br/>"
        Expr = self.expression(penetrated=True)
        Mappings = {iSymbol: iSymbol.name.replace("_", r"\_") for iSymbol in Expr.free_symbols}
        HTML += f"<b>表达式(穿透)</b>: {html.escape(Math(sympy.latex(Expr, symbol_names=Mappings, mul_symbol='times'))._repr_latex_())}<br/>"
        MetaData = self.getMetaData()
        MetaData = MetaData[~MetaData.index.str.contains("_QS")]
        HTML += f"<b>元信息</b>: {dict2html(MetaData)}"
        return HTML + super()._repr_html_()
    
    def equals(self, other):
        if self is other: return True
        if not isinstance(other, Factor): return False
        if not (isinstance(other, type(self)) or isinstance(self, type(other))): return False
        if not self._FactorTable.equals(other._FactorTable): return False
        if not (self._NameInFT != other._NameInFT): return False
        if self._QSArgs != other._QSArgs: return False
        if len(self.Descriptors) != len(other.Descriptors): return False
        for i, iDescriptor in enumerate(self.Descriptors):
            if iDescriptor != other.Descriptors[i]:
                return False
        return True


# 直接赋予数据产生的因子
# data: DataFrame(index=[时点], columns=[ID])
class DataFactor(Factor):
    class __QS_ArgClass__(Factor.__QS_ArgClass__):
        DataType = Enum("double", "string", "object", arg_type="SingleOption", label="数据类型", order=0, option_range=("double", "string", "object"))
        LookBack = Int(0, arg_type="Integer", label="回溯天数", order=1)
    
    def __init__(self, name, data, sys_args={}, config_file=None, **kwargs):
        if  isinstance(data, pd.Series):
            if data.index.is_all_dates:
                self._DataContent = "DateTime"
            else:
                self._DataContent = "ID"
            if "数据类型" not in sys_args:
                try:
                    data = data.astype(float)
                except:
                    sys_args["数据类型"] = "object"
                else:
                    sys_args["数据类型"] = "double"
        elif isinstance(data, pd.DataFrame):
            self._DataContent = "Factor"
            if "数据类型" not in sys_args:
                try:
                    data = data.astype(float)
                except:
                    sys_args["数据类型"] = "object"
                else:
                    sys_args["数据类型"] = "double"
        else:
            self._DataContent = "Value"
            if "数据类型" not in sys_args:
                if isinstance(data, str): sys_args["数据类型"] = "string"
                else:
                    try:
                        data = float(data)
                    except:
                        sys_args["数据类型"] = "object"
                    else:
                        sys_args["数据类型"] = "double"
        self._Data = data
        return super().__init__(name=name, ft=None, sys_args=sys_args, config_file=None, **kwargs)
    def getMetaData(self, key=None, args={}):
        DataType = args.get("数据类型", self._QSArgs.DataType)
        if key is None: return pd.Series({"DataType": DataType})
        elif key=="DataType": return DataType
        return None
    def getID(self, idt=None):
        if (self._OperationMode is not None) and (self._OperationMode._isStarted): return self._OperationMode.IDs
        if self._DataContent=="Factor":
            return self._Data.columns.tolist()
        elif self._DataContent=="ID":
            return self._Data.index.tolist()
        else:
            return []
    def getDateTime(self, iid=None, start_dt=None, end_dt=None):
        if (self._OperationMode is not None) and (self._OperationMode._isStarted): return self._OperationMode.DateTimes
        if self._DataContent in ("DateTime", "Factor"):
            return self._Data.index.tolist()
        else:
            return []
    def readData(self, ids, dts, **kwargs):
        if self._DataContent=="Value":
            return pd.DataFrame([(self._Data,)*len(ids)]*len(dts), index=dts, columns=ids)
        elif self._DataContent=="ID":
            Data = pd.DataFrame(self._Data.values.reshape((1, self._Data.shape[0])).repeat(len(dts), axis=0), index=dts, columns=self._Data.index)
        elif self._DataContent=="DateTime":
            Data = pd.DataFrame(self._Data.values.reshape((self._Data.shape[0], 1)).repeat(len(ids), axis=1), index=self._Data.index, columns=ids)
        else:
            Data = self._Data
        if Data.columns.intersection(ids).shape[0]==0:
            return pd.DataFrame(index=dts, columns=ids, dtype=("O" if self._QSArgs.DataType!="double" else float))
        if self._QSArgs.LookBack==0: return Data.reindex(index=dts, columns=ids)
        else: return fillNaByLookback(Data.reindex(index=sorted(Data.index.union(dts)), columns=ids), lookback=self._QSArgs.LookBack*24.0*3600).loc[dts, :]
    def __QS_prepareCacheData__(self, ids=None):
        return self._Data
    def _QS_getData(self, dts, pids=None, **kwargs):
        IDs = kwargs.get("ids", None)
        if IDs is None:
            IDs = self._OperationMode._FactorPrepareIDs[self.Name]
            if IDs is None:
                if pids is None:
                    IDs = list(self._OperationMode.IDs)
                else:
                    IDs = []
                    for iPID in pids: IDs.extend(self._OperationMode._PID_IDs[iPID])
            else:
                if pids is not None:
                    PrepareIDs = partitionListMovingSampling(IDs, len(self._OperationMode._PID_IDs))
                    IDs = []
                    for iPID in pids: IDs.extend(PrepareIDs[self._OperationMode._PIDs.index(iPID)])
        return self.readData(sorted(IDs), dts = list(dts))
    def equals(self, other):
        if self is other: return True
        if not super().equals(other): return False
        if self._DataContent != other._DataContent: return False
        if not (isinstance(self._Data, type(other._Data)) or isinstance(other._Data, type(self._Data))): return False
        if isinstance(self._Data, (pd.DataFrame, pd.Series)) and (not self._Data.equals(other._Data)): return False
        return (self._Data == other._Data)
