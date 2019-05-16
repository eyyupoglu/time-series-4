
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.tsa.api as smt

from statsmodels.graphics.tsaplots import plot_acf      # import ACF plot
from statsmodels.graphics.tsaplots import plot_pacf     # import PACF plot
from statsmodels.stats.diagnostic import acorr_ljungbox # import Ljung-Box Test
from statsmodels.tsa.arima_process import arma_generate_sample  # simulate ARMA process



def get_data(fileName):
    df = pd.read_csv(fileName + '.csv')
    df.index = df['t']
    df = df.drop(columns=['t'])
    return df


# Plot funtion to dsiplay both the time sereis, acf and pacf
# as deafualt it displays 25 lags,
# This code was adapted from the blog Seanabu.com
def tsplot(y, lags=25, figsize=(10, 8), style='bmh'):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    with plt.style.context(style):
        plt.figure(figsize=figsize)  # Set the size of the figure

        layout = (2, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))

        y.plot(ax=ts_ax)
        ts_ax.set_title('Time Series Analysis Plots')
        plot_acf(y, lags=lags, ax=acf_ax)
        plot_pacf(y, lags=lags, ax=pacf_ax, method='ywm')

        plt.tight_layout()

    return


# Plot function to displaythe standardized residuals, ACF and Ljung-Box test-
# statiticics p-values. As deafualt it displays 25 lags,
# This code was adapted from the blog Seanabu.com
def tsdiag(arimaResiduals, afcFags=25, lbLags=10, figsize=(10, 8), style='bmh'):
    if not isinstance(arimaResiduals, pd.Series):
        arimaFittedvVlues = pd.Series(arimaResiduals)

    with plt.style.context(style):
        plt.figure(figsize=figsize)  # Set the size of the figure

        layout = (3, 1)
        sr_ax = plt.subplot2grid(layout, (0, 0))
        acf_ax = plt.subplot2grid(layout, (1, 0))
        lb_ax = plt.subplot2grid(layout, (2, 0))

        # Create the standard residual plot
        sr_ax.plot(arimaFittedvVlues)
        sr_ax.set_title("Standardizede Residuals")
        sr_ax.set_xlabel("Time")

        # Crate the ACF plot
        plot_acf(arimaResiduals, lags=afcFags, ax=acf_ax)

        # Create the Ljung-Box statitics plot
        lb = acorr_ljungbox(arimaResiduals, lags=lbLags)
        lbPvalue = lb[1]  # get the pvalue from the ljungbox test

        lb_ax.scatter(np.arange(lbLags), lbPvalue, facecolors='none', edgecolors='b')
        lb_ax.set_ylim(-0.1, 1)
        lb_ax.axhline(y=0.05, linestyle='--')
        lb_ax.set_title("p values for Ljung-Box Statistic")
        lb_ax.set_ylabel("p values")
        lb_ax.set_xlabel("lags")

        plt.tight_layout()
    return


# The function apply the numpy function cumsum multiple times on the same
# the same array. Used for simulating the integrated part in ARIMA models.
def cusumRepeat(arma, d):
    if d == 0:
        return (arma)

    return (cusumRepeat(np.cumsum(arma), d - 1))


# Function to simulate ARIMA model as the python package statsmodels only
# supports simulations of ARMA models
def simArima(n,sigma, ar=np.array([0]), ma=np.array([0]),  d=0):
    if sigma == None:
        print('[WARNING] Sigma is set to 1!')
        sigma = 1
    if np.array_equal(ar, np.array([0])) and np.array_equal(ma, np.array([0])):
        print(
            "Neither autoregressive parameters and moving average parameters are set. At least one need to be speficifyed")
        return -1

    ar = np.r_[1, -ar]  # add zero-lag and negate
    ma = np.r_[1, ma]  # add zero-lag

    arimaSim = arma_generate_sample(ar, ma, n, sigma=sigma)

    # Apply integration to the arma simulation
    if d > 0:
        arimaSim = cusumRepeat(arimaSim, d)

    return arimaSim

if __name__ == '__main__':
    from pykalman import KalmanFilter
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv('data_building.csv')
    observed_inputs = df[['Ta', 'Ph', 'Gv']].values
    measurements = df['yTi'].values[:-3]
    test_measurements = df['yTi'].values[-3:]
    fig = plt.figure(figsize=(18, 16), dpi=80, facecolor='w', edgecolor='k')

    kf = KalmanFilter(transition_matrices=[[0.755, 0.24],  # A
                                           [0.1, 0.9]],
                      observation_matrices=[1, 0],  # C ???????????
                      initial_state_mean=[27, 23],
                      initial_state_covariance=[[0.5, 0],  # sigma
                                                [0, 0.5]],
                      observation_covariance=0.5,
                      transition_covariance=[[0.5, 0],
                                             [0, 0.5]],
                      em_vars=['transition_covariance', 'observation_covariance'])
    kf.em(measurements, n_iter=50)
    state_means, state_covariances = kf.filter(measurements)
    state_std = np.sqrt(state_covariances[:, 0])
    # print (state_std)
    # print (state_means)
    # print (state_covariances)

    plt.plot(measurements, '-r', label='measurement-Ti')
    plt.plot(state_means[:, 0], '-g', label='kalman-filter output- Ti')
    plt.plot(state_means[:, 1], '-b', label='kalman-filter output-Tm')
    plt.legend(loc='upper left', prop={'size': 15})
    plt.title('Kalman filter estimates', size=20)
    plt.show()