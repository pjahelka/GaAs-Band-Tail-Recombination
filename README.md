# GaAs Band-Tail Recombination Model

A GaAs solar cell modeller for p+/n devices that includes Urbach tails in the heavily doped emiiter.

The model includes for:
- Bandgap Narrowing
- Urbach tails in the heavily doped p-GaAs emitter
- Depletion region SRH recombination
- Surface Recombination

## Project Structure

- `main.py`: Contains the `GaAsCellCalculator` class, which performs the core physical simulations and current-voltage (IV) calculations.
- `config.py`: Central configuration file for physical constants (GaAs properties), device geometry, and simulation settings.
- `run_sweep.py`: A utility script to perform parameter sweeps over doping levels, SRV, and SRH lifetimes using multiprocessing.
- `plot_results.py`: Visualization script for generating contour plots and performance curves (Voc, efficiency, ideality factor) from simulation results.
- `results/`: Directory where simulation CSV files (IV curves and PV parameters) are stored.
- `plots/`: Directory where generated visualizations are saved.

## Installation

Ensure you have Python 3.8+ installed. The following libraries are required:

```bash
pip install numpy scipy matplotlib pandas
```
*(Optional)* `scienceplots` for enhanced plot styling.

## Usage

### Single Simulation
To run a single simulation with default parameters or specific values, use `main.py`:
```python
from main import GaAsCellCalculator

cell = GaAsCellCalculator(doping=1e19, srv=1e3, tau_SRH=1e-6)
cell.save_simulation_results()
```

### Parameter Sweep
To run a large-scale sweep across multiple variables:
```bash
python run_sweep.py
```
Adjust the grids in `run_sweep.py` to define the range of doping, SRV, and lifetimes you wish to explore.

### Plotting Results
After running a sweep, generate visualizations using:
```bash
python plot_results.py
```

## Physics Background
The model calculates the absorption coefficient $\alpha(E, \Delta\mu)$ by integrating over the distorted density of states in the heavily doped GaAs. It then uses the generalized Planck's law to determine the local radiative emission and integrates this over the device volume to find the total dark current. 
