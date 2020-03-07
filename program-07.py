#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 08:56:20 2020
This code is used to analysis the data about earthquakes for past 30 days 
The data was downloaded at the 8:57am 3/3/2020
@author: xu1361
"""
# Import necessary module
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import scipy as si
import numpy as np

# Read csv file
df = pd.read_table('all_month.csv', header=0, sep=',')

# Plot histogram of the magnitude of earthquakes
boundary = range(0, 11, 1)
plt.hist(df['mag'].dropna(), bins=boundary)
plt.xlabel('Magnitude')
plt.ylabel('Probability')
plt.title('Magnitude of earthquake')
plt.show()

# Plot KDE 
kde = stats.gaussian_kde(df['mag'].dropna())
kde.covariance_factor = lambda:0.1
kde._compute_covariance()
a = np.sort(df['mag'].dropna())
plt.plot(a, kde(a))
plt.xlabel('Magnitude')
plt.ylabel('Density')
plt.title('KDE Plot')
plt.show()
# The similarity of histogram and kde is that they both show the same distribution
# and they are tell us how probable it is to find a data point with a certain value
# The difference is that the distribution of kde is more smooth so that every 
# sample has a certain probability responding to it.

# Plot latitude vs longitude
plt.scatter(df['longitude'], df['latitude'])
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Distribution of earthquakes')
plt.show()
# From the plot we can see that most points located at the intersection 
# of continental plates

# Plot normalized cumulative distribution for depth
d = np.sort(df['depth'])
cdf = np.linspace(0, 1, len(d))
plt.plot(d, cdf)
plt.xlabel('Depth')
plt.ylabel('Cumulative Distribution')
plt.title('CDF Plot')
plt.show()
# Only a few earthquakes' depth can be bigger than 200. Most of thier depth are
# concentrated in 0 to 50.


# Plot scatter of magnitude with depth
plt.scatter(df['mag'], df['depth'])
plt.xlabel('Magnitude')
plt.ylabel('Depth')
plt.title('Magnitude VS Depth')
plt.show()
# The earthquakes with bigger magnitude are more possible to occur at deeper 
# positions

# Q-Q plot of magnitude
si.stats.probplot(df['mag'].dropna(), dist = 'norm', plot=plt)
plt.show()
