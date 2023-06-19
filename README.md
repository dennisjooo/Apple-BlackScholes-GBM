# Analysis of Apple’s Stock and Option
A simple analysis of Apple stock, calls, and puts using Black-Scholes and Geometric Brownian Motion. This was done for my mid-semester test on Computational Finance.

The stock data was taken from [Yahoo! Finance](https://finance.yahoo.com/quote/AAPL) from April 30th, 2021 up to April 29th, 2022, the analysis used the market closing price in particular. 
The Black-Scholes Option Pricing Model was used to simulate some trading strategies for calls and puts which expired on October 21st, 2022.
The recommended strategy using the simulated values for that day was a Long-Put Butterfly Spread using two $170-strike put, a $50-strike put, and a $210-strike put.

