import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.optimize import minimize


class Portfolio:
    TRADING_DAYS = int(252)

    def __init__(
        self,
        products_names: list,
        daily_prices: pd.DataFrame,
        risk_free: pd.DataFrame,
        weights: np.array = None,
        market: pd.DataFrame = None,
    ) -> None:
        self.products = products_names
        self.daily_prices = daily_prices
        self.daily_prices.index = self.daily_prices.index.tz_localize(None)
        self.risk_free = risk_free
        # equally weighted by default
        if weights is None:
            self.weights = np.array(len(self.products) * [1 / len(self.products)])
        else:
            self.weights = weights
        self.market = market

    def daily_returns(self):
        return (
            self.daily_prices.pct_change()
            .dropna(how="any")
            .merge(
                self.risk_free.divide(365),
                how="left",
                left_index=True,
                right_index=True,
            )
            .ffill()
        )

    def daily_excess_returns(self):
        daily_returns = self.daily_returns()
        for _c in daily_returns.columns:
            daily_returns[_c] = daily_returns[_c].sub(daily_returns["effective_rate"])
        return daily_returns.drop(["effective_rate"], axis=1)

    # def port_ret(self):
    #     mean = self.daily_returns().mean()
    #     return np.sum(mean * self.weights) * self.TRADING_DAYS

    def port_excess_ret(self):
        mean = self.daily_excess_returns().mean()
        return np.sum(mean * self.weights) * self.TRADING_DAYS

    def port_vol(self):
        cov = self.daily_excess_returns().cov()
        return np.sqrt(np.dot(self.weights.T, np.dot(cov, self.weights))) * np.sqrt(
            self.TRADING_DAYS
        )

    def sharpe_ratio(self):
        return self.port_excess_ret() / self.port_vol()

    def manual_input_weights(self):
        print("Fill in the current value of each asset (in $): ")
        values = []
        for p in self.products:
            v = input(p + ": ")
            if len(v) == 0:
                values.append(0)
            else:
                values.append(float(v))
        self.weights = np.array(values) / sum(values)
        self.info(sum(values))

    def beta(self):
        assert self.market is not None, "Market data is not provided."
        # TODO: merge market with excess returns
        self.market.columns = ["market"]
        daily_excess_returns = (
            self.daily_excess_returns()
            .merge(
                self.market,
                how="left",
                left_index=True,
                right_index=True,
            )
            .ffill()
        )
        # TODO: calculate individual betas
        betas = []
        for _s in daily_excess_returns.columns:
            if _s != "market":
                betas.append(
                    [
                        _s,
                        np.cov(
                            daily_excess_returns[_s], daily_excess_returns["market"]
                        )[0, 1]
                        / np.var(daily_excess_returns["market"]),
                    ]
                )
        return (
            pd.DataFrame(betas, columns=["ticker", "beta"])
            .set_index("ticker")
            .squeeze(axis=1)
        )

    def port_beta(self):
        return np.sum(self.beta() * self.weights)

    def info(self, total_value=None):
        print("Portfolio composition: ")
        if total_value is None:
            print(
                "{:^11s} {:>8s}  ".format(
                    "Products",
                    "Weights",
                )
            )
            for i in range(len(self.products)):
                print(
                    "{:^11s} {:>8.2%} ".format(
                        self.products[i],
                        self.weights.tolist()[i],
                    )
                )
        else:
            values = self.weights * total_value
            print("{:^11s} {:>8s}  {:>10s}".format("Products", "Weights", "Amounts"))
            for i in range(len(self.products)):
                print(
                    "{:^11s} {:>8.2%}  ${:>9,.2f}".format(
                        self.products[i], self.weights.tolist()[i], values[i]
                    )
                )
        print(
            "{:>25s}: {:>8.2%}".format("Expected excess return", self.port_excess_ret())
        )
        print("{:>25s}: {:>8.2%}".format("Standard deviation", self.port_vol()))
        print("{:>25s}: {:>8.2%}".format("Sharpe Ratio", self.sharpe_ratio()))
        print("{:>25s}: {:>8.4f}".format("Portfolio beta", self.port_beta()))


class EfficientFrontier(Portfolio):
    def __init__(
        self,
        products_names: list,
        daily_prices: pd.DataFrame,
        risk_free: float,
        weights=None,
        market=None,
        seed: np.int32 = 123,
        n: np.int32 = 10,
    ) -> None:
        super().__init__(products_names, daily_prices, risk_free, weights, market)
        self.seed = seed
        self.set_seed()
        self.n = n

    def set_seed(self):
        np.random.seed(self.seed)
        print("Set seed as", self.seed)

    def get_random_weights(self):
        r_weights = np.random.uniform(low=0, high=1, size=len(self.products))
        r_weights = r_weights / np.sum(r_weights)
        return r_weights

    def get_risk_return_spectrum(self):
        rr_list = []
        for _i in tqdm(range(self.n)):
            self.weights = self.get_random_weights()
            rr_list.append(
                [self.port_excess_ret(), self.port_vol(), self.sharpe_ratio()]
            )

        return pd.DataFrame(data=rr_list, columns=["return", "std", "sharpe_ratio"])


class EfficientPortfolio(Portfolio):
    def __init__(
        self,
        products_names: list,
        daily_prices: pd.DataFrame,
        risk_free: float,
        weights=None,
        market=None,
    ) -> None:
        super().__init__(products_names, daily_prices, risk_free, weights, market)
        self.efficiency: str = "Not optimized"
        # optimized weights are zeros before optimization
        self.optimized_weights = np.zeros(len(self.products))

    def _minus_SR(self, weights):
        self.weights = weights
        return -self.sharpe_ratio()

    def _port_vol_opt(self, weights):
        self.weights = weights
        return self.port_vol()

    def maximiaze_SR(self):
        # constraints, sum of weights = 1
        cons = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
        # bounds on weights: [0, 1]
        bnds = tuple((0, 1) for x in range(len(self.products)))
        # initial guess
        init_w = self.weights
        opts = minimize(
            self._minus_SR, init_w, method="SLSQP", bounds=bnds, constraints=cons
        )
        self.optimized_weights = opts.x
        self.efficiency = "Maximized Sharpe Ratio"

    def minimize_sd(self):
        # constraints, sum of weights = 1
        cons = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
        # bounds on weights: [0, 1]
        bnds = tuple((0, 1) for x in range(len(self.products)))
        # initial guess
        init_w = self.weights
        opts = minimize(
            self._port_vol_opt, init_w, method="SLSQP", bounds=bnds, constraints=cons
        )
        self.optimized_weights = opts.x
        self.efficiency = "Minimized volatility"


if __name__ == "__main__":
    import yfinance as yf
    from datetime import datetime, timedelta
    from interest_rate_fetcher import get_effective_rate

    tickers = [
        "QQQ",
        "VHT",  # Vanguard Health Care Index Fund ETF Shares
        "XLE",  # The Energy Select Sector SPDR Fund
        "VCR",  # Vanguard Consumer Discretionary ETF
        "VPU",  # Vanguard Utilities ETF
        "SHV",  # iShares Short Treasury Bond ETF
        "TLT",  # iShares 20+ Year Treasury Bond
    ]
    # market = ["^GSPC"]

    # set the end date to today
    end_date = datetime.today()
    # set the start date to 5 years ago
    start_date = end_date - timedelta(days=5 * 365)

    etf_df = yf.download(tickers, start=start_date, end=end_date)["Adj Close"]
    market = yf.download("^GSPC", start=start_date, end=end_date)[
        "Adj Close"
    ].to_frame()
    market.index = market.index.tz_localize(None)
    market = market.pct_change().dropna(how="any")

    etf_port = Portfolio(
        tickers,
        etf_df,
        get_effective_rate(start_date),
        market=market,
    )

    # etf_port.manual_input_weights()
    etf_port.info()
