# Data

# Data Format


## Data Code

### `load_syndata.py`

A python script that normalizes the synthetic data loaded from the MATLAB format (`.mat`). The script
will automatically find the characterisitc scale of each variable in the synthetic data, including 
spatial coordiates $(x,y)$, velocity $(u, v)$ and thickness $h$, and normalize them to be within
$[-1, 1]$. The script also re-organizes and reshapes the data to meet the requirement for the PINN
training. 
