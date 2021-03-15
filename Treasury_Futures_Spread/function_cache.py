import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy import stats
import sys



def find_next_contract(contract, contract_list):

	contract_sr = pd.Series(index=contract_list)

	if contract in contract_list:
		return contract_sr[contract:].index[1]
	else:
		return None

def convert_to_intention_day(roll_period, wind_trading_day, last_record):
	
	# get the intention day(last day of roll period) of each contract 
	dominant_contract = pd.DataFrame(roll_period["end"].shift(-1))
	dominant_contract.columns = ["intention_day"]
	dominant_contract = dominant_contract.astype("datetime64[ns]")

	num_of_nan = dominant_contract.isna().sum().values[0]
	# delete future contracts
	if num_of_nan > 1:
		dominant_contract = dominant_contract.iloc[: (-1) * (num_of_nan - 1), :]
	dominant_contract = dominant_contract.fillna(last_record)
	# get the next date(notice day) after intention day
	dominant_contract["notice_day"] = [wind_trading_day.loc[x:].index[1] for x in dominant_contract.iloc[:, 0]]
	dominant_contract = dominant_contract.astype(str)

	return dominant_contract


def treasury_futures_contracts_sequence(delivery_info):
	
	active_contracts = delivery_info.index.to_series()[:-1].values.reshape(-1, 1)
	deferred_contracts = delivery_info.index.to_series()[1:].values.reshape(-1, 1)
	# further contract is the next contract after deferred contract
	further_contracts = np.append(delivery_info.index.to_series().values[2:], None).reshape(-1, 1)
	
	notice_day = delivery_info["notice_day"].values[:-1].reshape(-1, 1)
	# gather info
	active_deferred_contracts = np.concatenate([active_contracts, deferred_contracts, further_contracts, notice_day], axis=1)
	# create dataframe
	contracts_sequence_df = pd.DataFrame(
		data=active_deferred_contracts,
		index=delivery_info["intention_day"][:-1],
		columns=[
			"active_contract",
			"deferred_contract",
			"further_contract",
			"notice_day",
		]
	)

	return contracts_sequence_df



def spread_change_during_roll(spread_price, roll_period, contract_list):

	# roll period from the aspect of active contract
	roll_period_for_active = roll_period.shift(-1).dropna()
	roll_period_for_active["spread_change"] = np.nan

	for active_contract in roll_period_for_active.index:
		# the deferred contract is the next contract after active contract
		deferred_contract = find_next_contract(active_contract, contract_list)
		start_date = roll_period_for_active.loc[active_contract, "start"]
		end_date = roll_period_for_active.loc[active_contract, "end"]
		# deferred contract becomes active after the roll period
		assert spread_price.loc[start_date, "active_contract"] == active_contract
		assert spread_price.loc[start_date, "deferred_contract"] == deferred_contract
		assert spread_price.loc[end_date, "active_contract"] == active_contract
		assert spread_price.loc[end_date, "deferred_contract"] == deferred_contract

		# spread change is the change of difference between acitve price and deferred price during the roll period
		spread_change = (spread_price.loc[end_date, "spread_price"] - spread_price.loc[start_date, "spread_price"])
		roll_period_for_active.loc[active_contract, "spread_change"] = spread_change
		spread_price.loc[start_date:end_date, "spread_price"].plot()
		plt.title(active_contract)
		plt.show()

	return pd.DataFrame(roll_period_for_active["spread_change"])


# get continuous data of the assigned value
def get_value_df(contracts_sequence, value, value_name, contract_list):

	value_df = pd.DataFrame(
		columns=[
			"active_contract",
			"deferred_contract",
			"further_contract",
			"active_contract_" + value_name,
			"deferred_contract_" + value_name,
			"further_contract_" + value_name,
		]
	)
	# get the notice day of last active contract
	# if last active contract is the first contract, then the last notice day is the first trading day
	for i in range(len(contracts_sequence)):
		if i == 0:
			last_notice_day = value.index[0]
		else:
			last_notice_day = contracts_sequence["notice_day"][i - 1]

		intention_day = contracts_sequence.index[i]
		active_contract_code = contracts_sequence["active_contract"][i]
		deferred_contract_code = contracts_sequence["deferred_contract"][i]
		further_contract_code = find_next_contract(deferred_contract_code, contract_list)
		
		# if the current active contract is not the last active contract, the contract is active from the last notice day to intention day
		if pd.to_datetime(intention_day) <= value.index[-1]:
			current_active_contract_value = value.loc[last_notice_day:intention_day, active_contract_code]
			current_deferred_contract_value = value.loc[last_notice_day:intention_day, deferred_contract_code]
			current_further_contract_value = value.loc[last_notice_day:intention_day, further_contract_code]
			active_contract_code_series = pd.Series(data=active_contract_code, index=current_active_contract_value.index)
			deferred_contract_code_series = pd.Series(data=deferred_contract_code, index=current_deferred_contract_value.index)
			further_contract_code_series = pd.Series(data=further_contract_code, index=current_further_contract_value.index)
		# if the last active contract in contracts sequence is the current active contract, the contract is active since the last notice day
		else:
			current_active_contract_value = value.loc[last_notice_day:, active_contract_code]
			current_deferred_contract_value = value.loc[last_notice_day:, deferred_contract_code]
			current_further_contract_value = value.loc[last_notice_day:, further_contract_code]

			if current_active_contract_value is not None: # check if there is any record of the last contract
				active_contract_code_series = pd.Series(data=active_contract_code, index=current_active_contract_value.index)
				deferred_contract_code_series = pd.Series(data=deferred_contract_code,index=current_deferred_contract_value.index)
				further_contract_code_series = pd.Series(data=further_contract_code,index=current_further_contract_value.index)

		current_data = pd.concat(
			[
				active_contract_code_series,
				deferred_contract_code_series,
				further_contract_code_series,
				current_active_contract_value,
				current_deferred_contract_value,
				current_further_contract_value,
			],
			axis=1,
		)
		current_data.columns = value_df.columns
		value_df = pd.concat([value_df, current_data], axis=0)

	last_intention_date = pd.to_datetime(intention_day)
	# if the last contract is not the current active contract, that means the current contract is not in the active contract list
	# add information about the current active contract and corresponding deferred contract
	if last_intention_date < value.index[-1]:

		current_contract = contracts_sequence.loc[intention_day, "deferred_contract"]
		next_contract = find_next_contract(current_contract, contract_list)
		further_contract = find_next_contract(next_contract, contract_list)

		latest_active_contract_value = value.loc[last_intention_date:, current_contract].iloc[1:]
		latest_deferred_contract_value = value.loc[last_intention_date:, next_contract].iloc[1:]
		latest_further_contract_value = value.loc[last_intention_date:, further_contract].iloc[1:]

		active_contract_code_series = pd.Series(data=current_contract, index=latest_active_contract_value.index)
		deferred_contract_code_series = pd.Series(data=next_contract, index=latest_deferred_contract_value.index)
		further_contract_code_series = pd.Series(data=further_contract, index=latest_further_contract_value.index)

		current_data = pd.concat(
			[
				active_contract_code_series,
				deferred_contract_code_series,
				further_contract_code_series,
				latest_active_contract_value,
				latest_deferred_contract_value,
				latest_further_contract_value,
			],
			axis=1,
		)
		current_data.columns = value_df.columns

		value_df = pd.concat([value_df, current_data], axis=0)

	value_df.index.name = "Trading_Day"

	return value_df


# add new feature to the feature dataframe, if the feature already exists, replace it with the new value
def add_feature(features, new_feature):
	if new_feature.columns[0] not in features.columns:
		features = pd.concat([features, new_feature], axis=1)
	else:
		features = features.drop(new_feature.columns[0], axis=1)
		features = pd.concat([features, new_feature], axis=1)
		
	return features

# calculate the spread change right before roll period of the assigned length
def spread_change_before_roll(days, spread_price, roll_period, contract_list):
	
	roll_period_for_active = roll_period.shift(-1).dropna()
	roll_period_for_active[str(days) + "d_spread_change"] = np.nan

	for active_contract in roll_period_for_active.index:

		deferred_contract = find_next_contract(active_contract, contract_list)
		# the start date of the roll period, in this case, it is the last date of the considered period
		start_date = roll_period_for_active.loc[active_contract, "start"]
		# the begin date of the considered period, days means days before the roll period
		begin_date = spread_price.loc[:start_date, :].index[-1 * days]

		assert spread_price.loc[begin_date, "active_contract"] == active_contract
		assert spread_price.loc[begin_date, "deferred_contract"] == deferred_contract
		assert spread_price.loc[start_date, "active_contract"] == active_contract
		assert spread_price.loc[start_date, "deferred_contract"] == deferred_contract
		# the change of spread during the period
		spread_change = (spread_price.loc[start_date, "spread_price"] - spread_price.loc[begin_date, "spread_price"])
		roll_period_for_active.loc[active_contract, str(days) + "d_spread_change"] = spread_change

	return pd.DataFrame(roll_period_for_active[str(days) + "d_spread_change"])


# calculate the exponential moving average 
def exp_mva(days, price, roll_period, contract_list, name):
	# including price on that day
	price_emva = price.ewm(span=days).mean()

	roll_period_for_active = roll_period.shift(-1).dropna()
	roll_period_for_active[str(days) + name] = np.nan

	for active_contract in roll_period_for_active.index:

		deferred_contract = find_next_contract(active_contract, contract_list)
		start_date = roll_period_for_active.loc[active_contract, "start"]

		roll_period_for_active.loc[active_contract, str(days) + name] = price_emva[start_date]

	return pd.DataFrame(roll_period_for_active[str(days) + name])



# correlation between two time series
def corr(days, var1, var2, roll_period, contract_list, name):

	roll_period_for_active = roll_period.shift(-1).dropna()
	roll_period_for_active[str(days) + name] = np.nan

	for active_contract in roll_period_for_active.index:

		deferred_contract = find_next_contract(active_contract, contract_list)
		last_date = roll_period.loc[active_contract, "end"]  # last date of last active period
		start_date = roll_period_for_active.loc[active_contract, "start"]  # start date of this roll period
		begin_date = var1[last_date:start_date].index[-1 * days]  # begin date of the considered period
		# calculate the correlation whenm the considered contract is active
		corr = var1[begin_date:start_date].corr(var2[begin_date:start_date])
		roll_period_for_active.loc[active_contract, str(days) + name] = corr

	return pd.DataFrame(roll_period_for_active[str(days) + name])


# calculate the n days standard deviation of the time series
# if divide is true, the result should be divided by its expanding standard deviation
def ndays_std(days, var, roll_period, contract_list, name, divide=True):
	
	roll_period_for_active = roll_period.shift(-1).dropna()
	roll_period_for_active[str(days)+name] = np.nan
	
	for active_contract in roll_period_for_active.index:      

		deferred_contract = find_next_contract(active_contract, contract_list)
		
		last_date = roll_period.loc[active_contract, "end"]# last date of last active period
		start_date = roll_period_for_active.loc[active_contract, 'start']# start date of this roll period
		begin_date = var[last_date:start_date].index[-1*days]# begin date of the considered period
		first_date = var[last_date:].index[1]# begin date of the expanding period
		# std of n days period 
		ndays_std = var[begin_date:start_date].std()
		
		if divide:
			ndays_std = ndays_std / var[first_date:start_date].std() # divide by the std of the expanding period
			
		roll_period_for_active.loc[active_contract, str(days)+name] = ndays_std
		
	return pd.DataFrame(roll_period_for_active[str(days)+name])


# calculate the current value of the considered variable
# if divide is true, the value is divided by n days moving average right before the rolling period
def current_value(days, var, roll_period, contract_list, name, divide=True):
	
	roll_period_for_active = roll_period.shift(-1).dropna()
	roll_period_for_active[str(days)+name] = np.nan
	
	for active_contract in roll_period_for_active.index:      

		deferred_contract = find_next_contract(active_contract, contract_list)

		start_date = roll_period_for_active.loc[active_contract, 'start']# start date of this roll period
		begin_date = var[:start_date].index[-1*days]# begin date of the considered period
		
		ndays_mva = var[begin_date:start_date].mean()# average value n days right before the roll period
		
		current_var = var[start_date]
		
		if divide:
			current_var = current_var / ndays_mva
			
		roll_period_for_active.loc[active_contract, str(days)+name] = current_var
		
	return pd.DataFrame(roll_period_for_active[str(days)+name])


def cal_zscore(var):
	return pd.Series((var - var.mean()) / var.std(), index=var.index)


# difference of two variables' z-scores
def Z_diff(days, var1, var2, roll_period, contract_list, name):
	
	roll_period_for_active = roll_period.shift(-1).dropna()
	roll_period_for_active[str(days)+name] = np.nan
	
	for active_contract in roll_period_for_active.index:      

		deferred_contract = find_next_contract(active_contract, contract_list)
		
		last_date = roll_period.loc[active_contract, "end"]# last date of last active period
		start_date = roll_period_for_active.loc[active_contract, 'start']# start date of this roll period
		begin_date = var1[last_date:start_date].index[-1*days]# begin date of the considered period
		
		z_diff = cal_zscore(var1)[begin_date:start_date] - cal_zscore(var2)[begin_date:start_date]# z-score difference
		
		roll_period_for_active.loc[active_contract, str(days)+name] = z_diff.mean()# average over the considered period
		
	return pd.DataFrame(roll_period_for_active[str(days)+name])

# calculate standard deviation of price
def cal_std(days, var, roll_period, contract_list):
	
	roll_period_for_active = roll_period.shift(-1).dropna()
	roll_period_for_active[str(days)+'d_std'] = np.nan
	
	for active_contract in roll_period_for_active.index:
		
		deferred_contract = find_next_contract(active_contract, contract_list)
		
		last_date = roll_period.loc[active_contract, "end"]# end date of last rolling period
		start_date = roll_period_for_active.loc[active_contract, 'start']# start date of this rolling period
		begin_date = var[last_date:start_date].index[-1*days]# begin date of the considered period
		
		std = var[begin_date:start_date].std()
		
		roll_period_for_active.loc[active_contract, str(days)+'d_std'] = std
		
	return pd.DataFrame(roll_period_for_active[str(days)+'d_std'])

def drawdown(nav):
	running_max = nav.expanding(min_periods=1).max()
	cur_dd = (nav - running_max)/running_max
	max_dd=-cur_dd.min()
	return max_dd

def median_skew(ret):
	return 3*(ret.mean()-ret.median())/ret.std()

def grading_(portfolio_ret, single=False, holding_days = 1):
	# the return column is daily return by default

	mul_factor = 252 / holding_days

	portfolio_ret = portfolio_ret.dropna()

	if len(portfolio_ret) == 0:
		return None, None, None

	if single:
		annual_return = portfolio_ret.mean() * mul_factor
	else:
		
		nav = (1+portfolio_ret).cumprod()
		annual_return = nav.iloc[-1]**(mul_factor/len(nav))-1
	
	annual_vol = portfolio_ret.std()*np.sqrt(mul_factor)
	sharpe_ratio = annual_return/annual_vol
	max_dd = drawdown(nav)
	normalized_drawdown=max_dd/annual_vol
	sr = (stats.mstats.gmean(1+portfolio_ret)-1)/portfolio_ret.std()
	skew = median_skew(portfolio_ret)
	adj_sr = sr/np.sqrt(1-portfolio_ret.skew()*sr+((portfolio_ret.kurtosis()-1)/4)*(sr**2))*np.sqrt(mul_factor)
	# Visualization
	nav.plot(legend=True)

	print('{0:.3f}'.format(sharpe_ratio), '{0:.3f}'.format(annual_return), '{0:.3f}'.format(annual_vol),\
		  '{0:.3f}'.format(max_dd), '{0:.3f}'.format(normalized_drawdown), '{0:.3f}'.format(skew), '{0:.3f}'.format(adj_sr))

	return annual_return, annual_vol, sharpe_ratio

def grading(portfolio_ret, single=False):
	holding_days = 1

	mul_factor = 252 / holding_days
	portfolio_ret = portfolio_ret.dropna()
	if len(portfolio_ret) == 0:
		return None, None, None

	if single:
		annual_return = portfolio_ret.mean() * mul_factor
	else:

		nav = (1+portfolio_ret).cumprod()
		annual_return = nav.iloc[-1]**(mul_factor/len(nav))-1

	annual_vol = portfolio_ret.std()*np.sqrt(mul_factor)
	sharpe_ratio = annual_return/annual_vol

	return annual_return, annual_vol, sharpe_ratio


from matplotlib import colors
def background_gradient(s, m, M, cmap='PuBu', low=0, high=0):
    rng = M - m
    norm = colors.Normalize(m - (rng * low),
                            M + (rng * high))
    normed = norm(s.values)
    c = [colors.rgb2hex(x) for x in plt.cm.get_cmap(cmap)(normed)]
    return ['background-color: %s' % color for color in c]



from scipy.stats import norm
data = rol_open_ret.dropna().values

# Fit a normal distribution to the data:
mu, std = norm.fit(data)

# Plot the histogram.
plt.hist(data, bins=40, density=True, alpha=0.6, color='g')

# Plot the PDF.
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
title = "Fit results: mu = %.5f,  std = %.5f" % (mu, std)
plt.title(title)

plt.show()

