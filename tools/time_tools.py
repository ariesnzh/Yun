import exchange_calendars as xcals

# 获取可用日历列表
def get_calendar_names(include_aliases: bool = False):
	return xcals.get_calendar_names(include_aliases=include_aliases)

# 判断某一天是否是交易日
def check_specific_day_is_trading_day(dt: str):
	xshg = xcals.get_calendar('XSHG')
	return xshg.is_session(dt)

# 获取指定日期前k个交易日列表(含传入的日期)
def get_k_trading_days_before(dt: str, k: int):
	xshg = xcals.get_calendar('XSHG')
	if not xshg.is_session(dt):
		dt = get_previous_trading_day(dt)
	return xshg.sessions_window(dt, -k)

# 获取指定日期后k个交易日列表(含传入的日期)
def get_k_trading_days_after(dt: str, k: int):
	xshg = xcals.get_calendar('XSHG')
	if not xshg.is_session(dt):
		dt = get_next_trading_day(dt)
	return xshg.sessions_window(dt, k)

# 获取某一天后的下一个交易日
def get_next_trading_day(dt: str):
	xshg = xcals.get_calendar('XSHG')
	if xshg.is_session(dt):
		next_trading_day = get_k_trading_days_after(dt, 2)[-1]
	else:
		next_trading_day = xshg.date_to_session(dt, direction='next')
	return next_trading_day

# 获取某一天前的前一个交易日
def get_previous_trading_day(dt: str):
	xshg = xcals.get_calendar('XSHG')
	return xshg.date_to_session(dt, direction='previous')


if __name__ == "__main__":
	print(check_specific_day_is_trading_day('2024-07-14'))
	print(get_k_trading_days_before('2024-07-12', 5))
	print(get_k_trading_days_after('2024-07-12', 5))
	print(get_next_trading_day('2024-07-12'))
	