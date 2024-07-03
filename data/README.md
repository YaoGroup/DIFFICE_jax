# Data

This folder contains several examples of remote-sensing velocity and thickness data for different 
ice shelves. The dataset for each ice shelf is directly cropped from the raw data files published online, 
maintaining the same resolution. No processing has been conducted on these data. The raw velocity data is downloaded
from NASA MEaSUREs Phase-Based Antarctica Ice Velocity Map, Version 1 [(NSIDC-0754)](https://nsidc.org/data/nsidc-0754/versions/1) 
with a 450m resolution. The thickness data is downloaded from NASA MEaSUREs BedMachine Antarctica, Version 2
[(NSIDC-0756)](https://nsidc.org/data/nsidc-0756/versions/2) with a 500m resolution.

Additionally, this folder provides the core codes that conduct the necessary preprocessing of the raw data,
including the normalization, reshaping and random sampling, which are all required to ensure a successful PINN
training. Two versions of codes are provided in this folder. The `pinns` folder involves the code is for the
regular PINN training,  and the `xpinns` folder is for the extended-PINN (XPINN) training.

## Data Format


## Code description

### `preprocessing.py`

Involving essential functions to normalize the raw data loaded from the MATLAB format (`.mat`). The 
function in the script will automatically detect the characterisitc scale of each variable in the 
dataset, including the spatial coordiates $(x,y)$, velocity $(u, v)$ and thickness $h$, and use them to 
normalize those variables to be within the range of $[-1, 1]$. The script also re-organizes and reshapes
the data that is required by the code in the `model` folder to ensure the success of the PINN training.


### `sampling.py`

A python script that normalizes the synthetic data loaded from the MATLAB format (`.mat`). The script
will automatically find the characterisitc scale of each variable in the synthetic data, including 
spatial coordiates $(x,y)$, velocity $(u, v)$ and thickness $h$, and normalize them to be within
$[-1, 1]$. The script also re-organizes and reshapes the data to meet the requirement for the PINN
training. 
