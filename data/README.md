# Data

This folder provides several examples of remote-sensing velocity and thickness data for different 
ice shelves. The dataset for each ice shelf is directly cropped from the raw data files published online, 
maintaining the same resolution. No processing has been conducted on these data. The raw velocity data is downloaded
from NASA MEaSUREs Phase-Based Antarctica Ice Velocity Map, Version 1 [(NSIDC-0754)](https://nsidc.org/data/nsidc-0754/versions/1) 
with a 450m resolution. The thickness data is downloaded from NASA MEaSUREs BedMachine Antarctica, Version 2
[(NSIDC-0756)](https://nsidc.org/data/nsidc-0756/versions/2) with a 500m resolution.

## Data Format


## Data Code

### `load_syndata.py`

A python script that normalizes the synthetic data loaded from the MATLAB format (`.mat`). The script
will automatically find the characterisitc scale of each variable in the synthetic data, including 
spatial coordiates $(x,y)$, velocity $(u, v)$ and thickness $h$, and normalize them to be within
$[-1, 1]$. The script also re-organizes and reshapes the data to meet the requirement for the PINN
training. 
