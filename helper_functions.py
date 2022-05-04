import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def geo_bm(data, drift, volatility, period, dt=1 / 252, seed=False, rep=1):
    """
    This function simulates next possible price using the Geometric Brownian Motion of an index or a stock
    This is also a vectorised implementation using NumPy, thus requiring NumPy and Pandas
    :param data: The data to simulate, expects a dataframe
    :param drift: The drift parameter, expected to be the annual mean of the geometric return or a number
    :param volatility: The volatility parameter, expected to be the annual standard deviation of the geometric return or
    a number
    :param period: The number of period ahead to simulate. Has to do with dt, e.g. if dt is 1/365, it will be in days or
    a number
    :param dt: The relative difference from t+1 and t, also determine the number of periods above or a number
    :param seed: A boolean to determine the usage of random seed in NumPy, allows replicable results
    :param rep: The amount of repetition done in the simulation, expects a number
    :return: A dataframe of the Geometric Brownian Motion done
    """

    # Checking if the seed parameter and setting the random seed if necessary
    if seed:
        np.random.seed(0)

    # Initiating an array of Ss with
    s = np.zeros(shape=(1, rep))
    s[0] = data[-1]

    # Finding the increment
    increment = np.exp(np.random.normal(drift * dt, volatility * np.sqrt(dt), size=(period - 1, rep)))

    # Combining the array of Ss and the increment
    s = np.concatenate((s, increment), axis=0)

    # Finding the cumulative product across the columns
    s = np.cumprod(s, axis=0)

    # Returning the result as a dataframe
    return pd.DataFrame(s)


def clean_option_data(filename):
    """
    This function will clean the option data taken from Yahoo Finance
    Requires Pandas to use
    :param filename: The file name of the option data saved, expects a string which represents a CSV file
    :return: The clean dataframe used during for our observation
    """
    # Importing call option data
    data = pd.read_csv(filename)

    # Renaming the columns
    data.rename(columns=str.lower, inplace=True)
    data.columns = data.columns.str.replace(' ', '_')

    # Dropping unnecessary columns
    data.drop(['unnamed:_0', 'contract_name', 'bid', 'ask', 'change',
               '%_change', 'volume', 'open_interest', 'implied_volatility'],
              axis=1,
              inplace=True)

    # Converting last_trade_date into datetime datatype
    data['last_trade_date'] = pd.to_datetime(data['last_trade_date'])

    # Returning the clean dataframe
    return data


def bs(data, s_0, kind, period, r, volatility):
    """
    This function calculates the modeled premium using the Black-Scholes option pricing model
    Technically, it should be used only on European style options
    Requires NumPy, Pandas, Copy, and SciPy
    :param data: The data used to run the model, expects a dataframe
    :param s_0: The latest stock price, expects a number
    :param kind: Expects either 'call' or 'put', indicates the which option and how to calculate it
    :param period: The number of period or distance between the observed date and the expiration date of the option,
    expects a number
    :param r: The annual continuous compounding interest rate, expects a number
    :param volatility: The implied volatility of the stock, can also be calculated from the standard deviation of the
    geometric return, expects a number
    :return: A dataframe of the original data along with the Black-Scholes modeled prices and whether the current price
    is over or undervalued
    """

    # Importing the normal distribution class  from SciPy
    from scipy.stats import norm

    # Importing the deepcopy function from copy
    from copy import deepcopy

    # Using deepcopy to copy the data
    data = deepcopy(data)

    # Calculating d1 and d2
    d1 = (np.log(s_0 / data['strike'].values) + (r + volatility ** 2 / 2) * period) / (volatility * np.sqrt(period))
    d2 = d1 - volatility * np.sqrt(period)

    # Checking whether it is a call option
    if str.lower(kind) == 'call':
        # Calculating Black-Scholes prices if it is a call option
        data['bs_price'] = s_0 * norm.cdf(d1) - data['strike'].values * np.exp(-r * period) * norm.cdf(d2)

    # Checking whether it is a put option
    elif str.lower(kind) == 'put':
        # Black-Scholes prices
        data['bs_price'] = data['strike'].values * np.exp(-r * period) * norm.cdf(-d2) - s_0 * norm.cdf(-d1)

    # Checking whether the option is overvalued or undervalued
    data['undervalued'] = data['bs_price'] > data['last_price']

    return data


def payoff_profit(data, expected_price, kind, r, period, option_type):
    """
    This function will calculate the payoff and the profit of your option based also in your position
    It will also convert the payoff and the cost relative the position, thus a short will have positive cost and
    negative payoff
    Requires Pandas, NumPy, and Copy
    :param data: The data which will be used to calculate the profit and payoff, expects a dataframe
    :param expected_price: The expected price at the expiration date, expects a number
    :param kind: Expects either 'call' or 'put', indicates the which option and how to calculate it
    :param r: The annual continuous compounding interest rate, expects a number
    :param period: The number of period or distance between the observed date and the expiration date of the option,
    expects a number
    :param option_type: Expects either 'long' or 'short, indicates the option's position
    :return: The dataframe with the added payoff and profit columns as well as converted cost column
    """

    # Importing the deepcopy function from copy
    from copy import deepcopy

    # Using deepcopy to copy our data
    data = deepcopy(data)

    # Setting the payoff columns to 0s
    data['payoff'] = 0

    # Iterating through the indexes in the data
    for i in range(len(data)):
        # Checking if the option kind is a call
        if str.lower(kind) == 'call':
            # Exercising the option relative to the expected price
            if data['strike'][i] >= expected_price:
                data['payoff'][i] = 0
            else:
                data['payoff'][i] = expected_price - data['strike'][i]

        # Checking if the option kind is a put
        elif str.lower(kind) == 'put':
            # Exercising the option relative to the expected price
            if data['strike'][i] <= expected_price:
                data['payoff'][i] = 0
            else:
                data['payoff'][i] = data['strike'][i] - expected_price

    # Checking whether the option is a long option and setting the payoff and cost appropriately
    if str.lower(option_type) == 'long':
        data['payoff'] = data['payoff']
        data['last_price_fv'] = -data['last_price'] * np.exp(r * period)
        data['bs_price_fv'] = -data['bs_price'] * np.exp(r * period)

    # Checking whether the option is a short option and setting the payoff and cost appropriately
    elif str.lower(option_type) == 'short':
        data['payoff'] = -data['payoff']
        data['last_price_fv'] = data['last_price'] * np.exp(r * period)
        data['bs_price_fv'] = data['bs_price'] * np.exp(r * period)

    # Dropping some redundant columns
    data.drop(['bs_price', 'last_price'], axis=1, inplace=True)

    # Calculating the profit using both the last trading price and the Black Scholes price
    data['bs_profit'] = data['payoff'] + data['bs_price_fv']
    data['profit'] = data['payoff'] + data['last_price_fv']

    # Returning the dataframe
    return data


def plot_payoff_profit(data, cutoff, title, save=False):
    """
    This function will plot a graph the payoff and the profit relative to the strike price at expiration
    Requires Pandas, Matplotlib, and Seaborn
    :param data: The data use to plot the graph, expects a dataframe
    :param cutoff: The cutoff point, also the expected stock price at expiration, expects a number
    :param title: The title for the plot which will be graphed, expects a string
    :param save: A boolean, if True, it will save the file
    """

    # Setting the figure size
    plt.figure(figsize=(8, 6))

    # Plotting the long call profit and payoff
    plt.plot(data['strike'], data[['payoff', 'bs_profit', 'profit']],
             label=['Payoff', 'Black Scholes Profit', 'Last Price Profit'])

    # Plotting the mean price at the expiration date
    plt.axvline(cutoff, color='red', alpha=0.5, label=f'Cut-off at {round(cutoff, 2)}')

    # Plot settings
    plt.title(title, fontsize=13, y=1.02)
    plt.xlabel('Strike Price ($)')
    plt.ylabel('Profit/Playoff ($)')
    plt.legend()
    sns.despine()

    # Saving the plot
    if save:
        plt.savefig('Graphs\\{name}.png'.format(name=title.lower().replace(r' ', '_')), transparent=True, dpi=500)

    # Showing the plot
    plt.show()


def pull_stock_yahoo_finance(index, start, stop):
    """
    This function will pull the stock price of a desired index from Yahoo Finance between a starting and ending dates
    Requires datetime, pandas_datareader, and pandas
    :param index: The desired index, expected to be a string
    :param start: The starting dates in string, expected format is MM-DD-YYYY
    :param stop: start: The ending dates in string, expected format is MM-DD-YYYY
    :return: A dataframe of the stock prices of your index
    """

    # Importing necessary functions
    from datetime import datetime
    from pandas_datareader.data import DataReader

    # Specify start date and end date
    end = datetime.strptime(stop, '%m-%d-%Y')
    start = datetime.strptime(start, '%m-%d-%Y')

    # Specify stocks that you want to get and data source
    ticker = [index]
    data_source = 'yahoo'

    # Read the data
    df = DataReader(ticker[0], data_source, start, end).to_frame()
    df.drop_duplicates(keep='first', inplace=True)


def pull_option_yahoo_finance(index, expiration):
    """
    This function will pull the available options from Yahoo Finance
    Requires Yahoo_fin and Pandas
    :param index: The desired index, expects a string
    :param expiration: The expiration date, expects a string in the format MM/DD/YYYY
    :return: Two dataframes of the calls and puts
    """

    # Importing the options function from yahoo_fin
    from yahoo_fin import options

    # Pulling the call options
    calls = options.get_calls(index, expiration)

    # Pulling the put options
    puts = options.get_puts(index, expiration)

    # Returning the dataframes
    return calls, puts
