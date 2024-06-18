from main import *
import numpy as np
import pandas as pd
import seaborn as sns




for quarter in np.arange(1,5,1):
    for year in np.arange(2020,2024):
        average = average_sentiments(quarter=quarter, year=year)
        df.append([quarter, year, average])



