# Uber stock dataset: knowledge base

## Dataset overview
The Uber stock dataset (uber_stock_data.csv) contains one row per trading day for Uber Technologies, Inc. (ticker UBER, New York Stock Exchange), from 2019-05-10 to 2025-02-05, for a total of 1,444 trading days. All prices are in US dollars per share. The columns are Date, Open, High, Low, Close, Adj Close and Volume.

## Date
Date is the trading day of the row, in YYYY-MM-DD format. Only days when the New York Stock Exchange was open appear: weekends and US market holidays are absent, so two consecutive rows can be several calendar days apart. For example, the first 10 calendar days of a month usually contain only 6 to 8 trading days.

## Open
Open is the opening price: the price of the first trade of the day, when the market opens at 9:30 a.m. Eastern Time. It reflects overnight news and sets how the session starts. Open can differ from the previous day's Close, which is called a gap up or a gap down.

## High
High is the intraday maximum: the highest price at which Uber shares traded at any moment during the session. On any given day, High is greater than or equal to Open, Low and Close.

## Low
Low is the intraday minimum: the lowest price at which Uber shares traded at any moment during the session. On any given day, Low is less than or equal to Open, High and Close.

## Close
Close is the closing price: the price of the last trade of the regular session, at 4:00 p.m. Eastern Time. It is the reference price for daily performance, returns and most stock charts.

## Adj Close
Adj Close is the adjusted closing price: the closing price corrected for corporate actions such as stock splits and dividend payments, so that prices from different periods can be compared fairly. In this dataset, Adj Close is identical to Close on every row, which means no split or dividend adjustment applies to Uber over this period.

## Volume
Volume is the trading volume: the number of Uber shares traded, bought and sold, during the day. It answers how many shares changed hands. It counts shares, not dollars. High volume signals strong market interest, often around earnings reports or major news. The median daily volume in this dataset is about 20 million shares, and the busiest days exceed 300 million shares.

## Daily range and volatility
The daily range is High minus Low: how much the price moved within a single trading session, from its lowest point to its highest point. It measures intraday movement. A wide range means a volatile day, a narrow range a calm one. Over a longer period, volatility is usually measured as the standard deviation of daily returns.

## Daily return
The daily return is the close-to-close percentage change: how much the closing price changed from one trading day to the next, computed as (Close today - Close previous day) / Close previous day.

## IPO
Uber went public on the New York Stock Exchange on 2019-05-10, which is the first row of the dataset. The IPO was priced at 45 dollars per share. On its first day the stock opened at 42 dollars and closed at 41.57 dollars, below its IPO price.