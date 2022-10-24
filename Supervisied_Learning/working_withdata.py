import os
import numpy as np      # works as Matlab and R
import pandas as pd     # worsk as excel with dataFrames and datasets
from turtle import color    
from matplotlib import pyplot as plt


def initialize():
    #read csv file from data file for windows we use from the first   
    CURRENT_DIR = os.path.dirname(__file__)
    file_path   = os.path.join(CURRENT_DIR, 'data/flights.csv')

    df_flights = pd.read_csv(file_path)
    df_flights.head()
    df_flights.isnull().sum()
    df_flights[df_flights.isnull().any(axis=1)]
    df_flights[df_flights.isnull().any(axis=1)].DepDelay.describe()

    #show all indexes
    df_flights.columns
    # fill all on null 
    df_flights.DepDel15 = df_flights.DepDel15.fillna(df_flights.DepDel15.mean())

    # Call the function for each delay field
    delayFields = ['DepDelay','ArrDelay']
    for col in delayFields:
        show_distribution(df_flights[col])



    # Trim outliers for ArrDelay based on 1% and 90% percentiles
    ArrDelay_01pcntile = df_flights.ArrDelay.quantile(0.01)
    ArrDelay_90pcntile = df_flights.ArrDelay.quantile(0.90)
    df_flights = df_flights[df_flights.ArrDelay < ArrDelay_90pcntile]
    df_flights = df_flights[df_flights.ArrDelay > ArrDelay_01pcntile]

    # Trim outliers for DepDelay based on 1% and 90% percentiles
    DepDelay_01pcntile = df_flights.DepDelay.quantile(0.01)
    DepDelay_90pcntile = df_flights.DepDelay.quantile(0.90)
    df_flights = df_flights[df_flights.DepDelay < DepDelay_90pcntile]
    df_flights = df_flights[df_flights.DepDelay > DepDelay_01pcntile]

    # View the revised distributions
    for col in delayFields:
        show_distribution(df_flights[col])
    
    print(df_flights.describe())

    print(df_flights[delayFields].mean())

    for col in delayFields:
        df_flights.boxplot(column=col, by='Carrier', figsize=(8,8))

    return  

def show_distribution(data_col):
    
    mean_value      = data_col.mean()
    min_value       = data_col.min()
    max_value       = data_col.max() 
    median_value    = data_col.median() 
    mode_value      = data_col.mode()[0]
    
    print('Minimum:{:.2f}\nMean:{:.2f}\nMedian:{:.2f}\nMode:{:.2f}\nMaximum:{:.2f}\n'.format(min_value,
                                                                                            mean_value,
                                                                                            median_value,
                                                                                            mode_value,
                                                                                            max_value))

    fig, ax = plt.subplots(2,1, figsize=(10,4))

    ax[0].hist(data_col)
    ax[0].set_ylabel('Frequency')

    # show the all value of mean, max, min in the histogram
    ax[0].axvline(x=min_value,      color ='gray',      linestyle='dashed', linewidth = 2)
    ax[0].axvline(x=mean_value,     color = 'cyan',     linestyle='dashed', linewidth = 2)
    ax[0].axvline(x=median_value,   color = 'red',      linestyle='dashed', linewidth = 2)
    ax[0].axvline(x=mode_value,     color = 'yellow',   linestyle='dashed', linewidth = 2)
    ax[0].axvline(x=max_value,      color = 'gray',     linestyle='dashed', linewidth = 2)

    ax[1].boxplot(data_col, vert=False)
    ax[1].set_xlabel('Values')

    fig.suptitle('Data Distribution:')

    fig.show()

    return True



if __name__ == "__main__":
    initialize()