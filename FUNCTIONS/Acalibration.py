import pandas as pd
from scipy.signal import savgol_filter

def apply_savgol(X):
    return savgol_filter(X, window_length=51, polyorder=3, axis=1)

SMOOTH = False

file = pd.read_csv('CSVfiles/uncalibrateddata.csv')
calibrations = pd.read_csv('CSVfiles/averagecalibrations2.csv')
columns = calibrations.drop('leaf_type', axis=1).columns
labels = file[['leaf_type','sample_id']]
values = file[columns]

cal_labels = calibrations[['leaf_type']]
cal_values = calibrations[columns]
if SMOOTH:
    smooth_values = apply_savgol(values.values)
    smooth_calv = apply_savgol(cal_values.values)
else:
    smooth_values = values.values
    smooth_calv = cal_values.values

file_smooth = pd.DataFrame(smooth_values,columns=columns)
cal_smooth = pd.DataFrame(smooth_calv, columns=columns)

file_smooth = file_smooth.clip(lower=1)
cal_smooth = cal_smooth.clip(lower=1)

file_smooth = pd.concat([labels, file_smooth], axis=1)
cal_smooth = pd.concat([cal_labels, cal_smooth], axis=1)


calibration_map = {'Calibration1': 'Gum_old','Calibration2':'Pine','Calibration3': 'Oakcork','Calibration4': 'Gum_young'}
cal_lookup = cal_smooth.set_index('leaf_type').rename(index=calibration_map)

def calibrate_row(row): 
    leaf_type = row['leaf_type']
    calibration_data = cal_lookup.loc[leaf_type]
    divided_data = row[columns] /calibration_data
    return divided_data

file_calibrated = file_smooth.copy()
file_calibrated[columns] = file_smooth.apply(calibrate_row, axis=1)


name = 'datacalibrated.csv'
file_calibrated.to_csv(name, index=False)
print(f"New file saved as: {name}")
