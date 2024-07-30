__all__ = ['deal_start_end_time']
import re
import datetime as dt


def deal_str_datetime(s: str):
    '''
    处理start 和 end
    '''
    time_pattern1 = re.compile('(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})')
    time_pattern2 = re.compile('(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})')
    time_pattern3 = re.compile('(\d{4})-(\d{2})-(\d{2})')
    time_pattern4 = re.compile('(\d{4})(\d{2})(\d{2})')

    re_list = [p.match(s) for p in [time_pattern1, time_pattern2, time_pattern3, time_pattern4]]

    if not sum(map(bool, re_list)):
        raise Exception(f'{s} 有问题')
    
    result = [resu for resu in re_list if bool(resu)][0].groups()
    result = [int(i) for i in result]
    if len(result) == 3:
        dat = dt.datetime(result[0],result[1],result[2])
    else:
        dat = '-'.join(result[:3])+' '+':'.join(result[-3:]) 
        dat = dt.datetime(result[0],result[1],result[2],result[-3],result[-2],result[-1])
    
    return dat


def deal_start_end_time(s: str, e:str):
    '''
    处理start 和 end
    '''
    time_pattern1 = re.compile('(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})')
    time_pattern2 = re.compile('(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})')
    time_pattern3 = re.compile('(\d{4})-(\d{2})-(\d{2})')
    time_pattern4 = re.compile('(\d{4})(\d{2})(\d{2})')
    
    date_list = []
    for v,dat in {dt.datetime(2010,1,1): s, dt.datetime.today(): e}.items():
        if not len(dat):
            date_list.append(v)
            continue
        re_list = [p.match(dat) for p in [time_pattern1, time_pattern2, time_pattern3, time_pattern4]]

        if not sum(map(bool, re_list)):
            raise Exception(f'{dat} 有问题')
        result = [resu for resu in re_list if bool(resu)][0].groups()
        result = [int(i) for i in result]
        if len(result) == 3:
            dat = dt.datetime(result[0],result[1],result[2])
        else:
            dat = '-'.join(result[:3])+' '+':'.join(result[-3:]) 
            dat = dt.datetime(result[0],result[1],result[2],result[-3],result[-2],result[-1])
        date_list.append(dat)

    return date_list[0], date_list[1]