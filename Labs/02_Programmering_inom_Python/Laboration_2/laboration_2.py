# import numpy as np

# data = np.loadtxt("C:\Users\hellg\OneDrive\Dokument\Python\python-programming-PATRIK-HELLGREN\Labs\02-Programmering-inom-Python\Laboration-2\datapoints.txt", delimiter=",")
# x = data[:, 0]
# y = data[:, 1]

# print(data)

import pandas as pd

df = pd.read_csv("\python-programming-PATRIK-HELLGREN\Labs\02_Programmering_inom_Python\Laboration_2\datapoints.txt")
print(df.head())
