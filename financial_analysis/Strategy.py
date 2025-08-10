<<<<<<< HEAD
import backtrader as bt
import pandas as pd
import numpy as np


class RsiEmaStrategy(bt.Strategy):
    params = (
        ('emaPeriod', 10),
        ('rsiPeriod', 14),
        ('buy_threshold', 1.005),
        ('sell_threshold', 0.995),
        ('rsi_buy', 40),
        ('rsi_sell', 60),
    )

    def __init__(self):
        self.ema = bt.indicators.EMA(self.data.close, period=self.p.emaPeriod)
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsiPeriod)

        self.prediction = 0
        self.uncertainty = 0

    def next(self):

        rsi = self.calculateRsi()
        macd = self.calculateMacd()
        volZ = self.calculateVolumeZscore()
        vol = self.calculateVolatility()
        overnightGap = self.calculateOvernightGap()
        ret1 = self.calculateReturn(1)
        ret3 = self.calculateReturn(3)
        ret5 = self.calculateReturn(5)

        row = {
            'date' : self.data.datetime.date(0),
            'close': self.data.close[0],
            'volume': self.data.volume[0],
            'open': self.data.open[0],
            'volume_zscore': volZ,
            'rsi': rsi,
            'macd': macd,
            'overnight_gap': overnightGap,
            'return_lag_1': ret1,
            'return_lag_3': ret3,
            'return_lag_5': ret5,
            'volatility': vol,
        }

        try:

            self.prediction = 0
            currentPrice = self.data.close[0]

            
            if rsi < self.p.rsi_buy and self.getposition().size == 0:
                #if self.prediction > currentPrice * self.p.buy_threshold: this line is for when we give you guys the prediction. 
                self.buy(size=10)
            elif rsi > self.p.rsi_sell and self.getposition().size > 0:
                #if aself.prediction < currentPrice * self.p.sell_threshold: this line again same thing
                self.close()

        except Exception as e:
            print(f"Trade error: {str(e)}")

    def calculateMacd(self):
        if len(self.data) < 26:
            return 0.0
        closes = np.array([self.data.close[-i] for i in range(26)][::-1])
        ema12 = closes[-12:].mean()
        ema26 = closes.mean()
        return ema12 - ema26

    def calculateVolumeZscore(self):
        if len(self.data) < 20:
            return 0.0
        volumes = np.array([self.data.volume[-i] for i in range(20)][::-1])
        currentVolume = volumes[-1]
        meanVolume = volumes.mean()
        stdVolume = volumes.std() + 1e-6
        return (currentVolume - meanVolume) / stdVolume

    def calculateVolatility(self):
        if len(self.data) < 20:
            return 0.0
        closes = np.array([self.data.close[-i] for i in range(20)][::-1])
        returns = np.diff(closes) / closes[:-1]
        return returns.std()

    def calculateOvernightGap(self):
        if len(self.data) < 2:
            return 0.0
        prevClose = self.data.close[-1]
        currentOpen = self.data.open[0]
        return np.log(currentOpen / prevClose)

    def calculateReturn(self, lag):
        if len(self.data) < lag + 1:
            return 0.0
        currentClose = self.data.close[0]
        pastClose = self.data.close[-lag]
        return (currentClose / pastClose) - 1

    def calculateRsi(self):
        if len(self.data) < 15:
            return 0.0
        closes = np.array([self.data.close[-i] for i in range(15)][::-1])
        deltas = np.diff(closes)
        gains = deltas.clip(min=0)
        losses = -deltas.clip(max=0)
        avgGain = gains.mean()
        avgLoss = losses.mean() + 1e-10
        rs = avgGain / avgLoss
        return 100 - (100 / (1 + rs))
=======
import backtrader as bt
import pandas as pd
import numpy as np


class RsiEmaStrategy(bt.Strategy):
    params = (
        ('emaPeriod', 10),
        ('rsiPeriod', 14),
        ('buy_threshold', 1.005),
        ('sell_threshold', 0.995),
        ('rsi_buy', 40),
        ('rsi_sell', 60),
    )

    def __init__(self):
        self.ema = bt.indicators.EMA(self.data.close, period=self.p.emaPeriod)
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsiPeriod)

        self.prediction = 0
        self.uncertainty = 0

    def next(self):

        rsi = self.calculateRsi()
        macd = self.calculateMacd()
        volZ = self.calculateVolumeZscore()
        vol = self.calculateVolatility()
        overnightGap = self.calculateOvernightGap()
        ret1 = self.calculateReturn(1)
        ret3 = self.calculateReturn(3)
        ret5 = self.calculateReturn(5)

        row = {
            'date' : self.data.datetime.date(0),
            'close': self.data.close[0],
            'volume': self.data.volume[0],
            'open': self.data.open[0],
            'volume_zscore': volZ,
            'rsi': rsi,
            'macd': macd,
            'overnight_gap': overnightGap,
            'return_lag_1': ret1,
            'return_lag_3': ret3,
            'return_lag_5': ret5,
            'volatility': vol,
        }

        try:

            self.prediction = 0
            currentPrice = self.data.close[0]

            
            if rsi < self.p.rsi_buy and self.getposition().size == 0:
                #if self.prediction > currentPrice * self.p.buy_threshold: this line is for when we give you guys the prediction. 
                self.buy(size=10)
            elif rsi > self.p.rsi_sell and self.getposition().size > 0:
                #if aself.prediction < currentPrice * self.p.sell_threshold: this line again same thing
                self.close()

        except Exception as e:
            print(f"Trade error: {str(e)}")

    def calculateMacd(self):
        if len(self.data) < 26:
            return 0.0
        closes = np.array([self.data.close[-i] for i in range(26)][::-1])
        ema12 = closes[-12:].mean()
        ema26 = closes.mean()
        return ema12 - ema26

    def calculateVolumeZscore(self):
        if len(self.data) < 20:
            return 0.0
        volumes = np.array([self.data.volume[-i] for i in range(20)][::-1])
        currentVolume = volumes[-1]
        meanVolume = volumes.mean()
        stdVolume = volumes.std() + 1e-6
        return (currentVolume - meanVolume) / stdVolume

    def calculateVolatility(self):
        if len(self.data) < 20:
            return 0.0
        closes = np.array([self.data.close[-i] for i in range(20)][::-1])
        returns = np.diff(closes) / closes[:-1]
        return returns.std()

    def calculateOvernightGap(self):
        if len(self.data) < 2:
            return 0.0
        prevClose = self.data.close[-1]
        currentOpen = self.data.open[0]
        return np.log(currentOpen / prevClose)

    def calculateReturn(self, lag):
        if len(self.data) < lag + 1:
            return 0.0
        currentClose = self.data.close[0]
        pastClose = self.data.close[-lag]
        return (currentClose / pastClose) - 1

    def calculateRsi(self):
        if len(self.data) < 15:
            return 0.0
        closes = np.array([self.data.close[-i] for i in range(15)][::-1])
        deltas = np.diff(closes)
        gains = deltas.clip(min=0)
        losses = -deltas.clip(max=0)
        avgGain = gains.mean()
        avgLoss = losses.mean() + 1e-10
        rs = avgGain / avgLoss
        return 100 - (100 / (1 + rs))
>>>>>>> 100812857436ca93a1602d0ff9d269777f2db7a4
