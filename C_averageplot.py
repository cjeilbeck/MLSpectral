import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from A_functions import moving_average, indicesmav, problemtype, read_data

plt.rcParams.update({'font.size': 25})
SAVGOL = True

MOV = False
HAIRCUT = True
start= 200
end = 2800

INDICES = False
if INDICES:
    HAIRCUT = False
    SAVGOL = False
    MOV = False

OAK = False
BINARY = False
GUMONLY = False
ALL = True

if OAK: problem = 'oak'
if GUMONLY: problem = 'gum'
if BINARY: problem = 'binary'
if ALL: problem = 'all'
filename = 'CSVfiles/datacalibrated.csv'
x,y = read_data(filename)
data = x.copy()
data['leaf_type']=y
data = problemtype(data, problem)
data_avg = data.groupby('leaf_type').mean()

plt.figure(figsize=(12, 7))

wavelengths = [float(col_name) for col_name in data_avg.columns]


if HAIRCUT:
    x_plot = wavelengths[start:end]
else:
    x_plot = wavelengths

if INDICES:
    x_plot = indicesmav(x_plot)

for leaf_type in data_avg.index:
    chosenspectra = data_avg.loc[leaf_type]
    y_counts = chosenspectra.values
    
    if SAVGOL:
        y_counts = savgol_filter(y_counts,window_length=51, polyorder=3)
    if MOV:
        y_counts = moving_average(y_counts,51)
    if HAIRCUT:
        y_plot = y_counts[start:end]
    else:
        y_plot = y_counts

    plt.plot(x_plot, y_plot, label=f"{leaf_type}")

plt.xlabel("Wavelength (nm)")
plt.ylabel("Reflectance")
plt.legend()
plt.show()


calibrations = pd.read_csv("CSVfiles/averagecalibrations2.csv", index_col=0)


wavelengths = [float(col_name) for col_name in x.columns]

plt.figure(figsize=(12, 7))


x_plot = wavelengths[start:end]
              



for leaf_type in calibrations.index:
    y_counts = calibrations.loc[leaf_type].values
    if SAVGOL:
        y_counts = savgol_filter(y_counts,window_length=51, polyorder=3)
    
    y_plot = y_counts[start:end]
    
    plt.plot(x_plot, y_plot, label=f"{leaf_type}")

plt.xlabel("Wavelength (nm)")
plt.ylabel("Reflectance")
plt.legend()
plt.show()