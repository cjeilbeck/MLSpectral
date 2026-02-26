import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from A_functions import moving_average, indicesmav

plt.rcParams.update({'font.size': 14})
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

filename = 'CSVfiles/datacalibrated.csv'

file = pd.read_csv(filename, header=0)
file = file.set_index(['leaf_type', 'sample_id'])
file_avg = file.groupby(level='leaf_type').mean()


name_mapping = {
    'Oakcork': 'Oak',
    'Gum_old': 'Old Gum',
    'Gum_young': 'Young Gum',
    'Pine': 'Pine'
}

file_avg = file_avg.rename(index=name_mapping)
file = file.rename(index=name_mapping)

plt.figure(figsize=(12, 7))

wavelengths = [float(col_name) for col_name in file.columns]


if HAIRCUT:
    x_plot = wavelengths[start:end]
else:
    x_plot = wavelengths

if INDICES:
    x_plot = indicesmav(x_plot)

for leaf_type in file_avg.index:
    chosenspectra = file_avg.loc[leaf_type]
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
plt.ylabel("Corrected Reflectance")
plt.legend()
plt.show()
