import pandas as pd
from pathlib import Path

directory = Path('.') 
skip = 14
ignore = ['Calibration1','Calibration2','Calibration3','Calibration4'] 

all_data = []
for classfolder in directory.iterdir():
    if classfolder.name not in ignore:
        leaf_type = classfolder.name
        for file in classfolder.glob('*.txt'):   
            filename = file.name
            part = filename.split('__')[1]
            sample_id = part.split('_')[0]
            data = pd.read_csv(file,skiprows=skip,header=None,names=['Wavelength','Counts'],delim_whitespace=True)
            data['leaf_type'] = leaf_type 
            data['sample_id'] = f"{leaf_type}_{sample_id}" 
            all_data.append(data)

combined = pd.concat(all_data, ignore_index=True)  
rot_data = combined.pivot_table(index=['leaf_type', 'sample_id'],columns='Wavelength',values='Counts')
rot_data = rot_data.reset_index()
filename = 'test.csv'
rot_data.to_csv(filename, index=False)



