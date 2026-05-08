import numpy as np
import os
import csv
from main import GaAsCellCalculator
import config as cfg
import multiprocessing

def run_simulation(params):
    doping, srv, tau_srh = params
    try:
        cell = GaAsCellCalculator(doping=doping, srv=srv, tau_SRH=tau_srh)
        pv = cell.save_simulation_results(verbose=False)
        pv['doping'] = doping
        pv['srv'] = srv
        pv['tau_srh'] = tau_srh
        return pv
    except Exception as e:
        print(f"Error for doping={doping:.1e}, srv={srv:.1e}, tau={tau_srh:.1e}: {e}")
        return None

def main():
    # Define Sweep Parameters
    # Doping grid
    doping_start = 1e19
    doping_end = 1e20
    doping_points = 10

    # SRV grid
    srv_start = 1e3
    srv_end = 1e10
    srv_points = 8

    # Tau_SRH grid
    tau_start = 1e-10
    tau_end = 100e-6
    tau_points = 7

    #doping_grid = np.linspace(doping_start, doping_end, doping_points)
    doping_grid = [5E19]
    #srv_grid = np.logspace(np.log10(srv_start), np.log10(srv_end), srv_points)
    srv_grid = [1E8, 1E100]
    tau_grid = np.logspace(np.log10(tau_start), np.log10(tau_end), tau_points)

    sweep_params = []
    for d in doping_grid:
        for s in srv_grid:
            for t in tau_grid:
                sweep_params.append((d, s, t))

    print(f"Starting sweep with {len(sweep_params)} points...")

    results = []
    # Sequential run because nested multiprocessing might fail or be inefficient
    for i, params in enumerate(sweep_params):
        print(f"Running simulation {i+1}/{len(sweep_params)}: doping={params[0]:.1e}, srv={params[1]:.1e}, tau={params[2]:.1e}")
        res = run_simulation(params)
        if res:
            results.append(res)

    # Filter out None results
    results = [r for r in results if r is not None]

    if not results:
        print("No results generated.")
        return

    # Create results folder
    results_dir = cfg.RESULTS_DIR
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    output_file = os.path.join(results_dir, "sweep_results.csv")
    
    # Write to CSV
    keys = results[0].keys()
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

    print(f"Sweep complete. Results saved to {output_file}")

if __name__ == "__main__":
    main()
