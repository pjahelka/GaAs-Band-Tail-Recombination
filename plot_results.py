import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.interpolate import interp1d
import config as cfg

try:
    import scienceplots
    plt.style.use(['science', 'notebook'])
except ImportError:
    pass

def plot_ideality_contours(csv_path='results/sweep_results.csv', output_dir='plots'):
    """
    Reads sweep results and generates contour plots of ideality factor vs SRV and Tau_SRH
    for each unique doping level.
    """
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Please run the simulation sweep first.")
        return

    # Load data
    df = pd.read_csv(csv_path)
    
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    doping_levels = sorted(df['doping'].unique())
    print(f"Found {len(doping_levels)} unique doping levels.")

    for doping in doping_levels:
        df_sub = df[df['doping'] == doping]
        
        # Pivot the data to create a grid for contour plotting
        # index: Y-axis (tau_srh), columns: X-axis (srv), values: Z-axis (ideality)
        try:
            pivot_df = df_sub.pivot(index='tau_srh', columns='srv', values='ideality')
        except Exception as e:
            print(f"Error pivoting data for doping {doping:.1e}: {e}")
            continue
            
        X = pivot_df.columns.values
        Y = pivot_df.index.values
        Z = pivot_df.values
        
        # Create the plot
        plt.figure(figsize=(10, 8))
        
        # Use log levels for better contour distribution if needed, 
        # but ideality factor is usually in a small linear range (1 to 2.5)
        levels = np.linspace(np.nanmin(df['ideality']), np.nanmax(df['ideality']), 21)
        
        cp = plt.contourf(X, Y, Z, levels=levels, cmap='RdYlBu_r')
        cbar = plt.colorbar(cp)
        cbar.set_label('Ideality Factor (n)', fontsize=12)
        
        # Set scales to log as SRV and Tau usually span orders of magnitude
        plt.xscale('log')
        plt.yscale('log')
        
        plt.xlabel('Surface Recombination Velocity (SRV) [cm/s]', fontsize=12)
        plt.ylabel('SRH Lifetime ($\\tau_{SRH}$) [s]', fontsize=12)
        plt.title(f'Ideality Factor Contour\nDoping = {doping:.1e} $cm^{-3}$', fontsize=14)
        
        # Add labels to contours
        # CS = plt.contour(X, Y, Z, levels=levels, colors='k', linewidths=0.5)
        # plt.clabel(CS, inline=True, fontsize=8, fmt='%.2f')

        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.tight_layout()
        
        # Save plot
        safe_doping_str = f"{doping:.1e}".replace('+', '')
        filename = os.path.join(output_dir, f'ideality_contour_doping_{safe_doping_str}.png')
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"Saved: {filename}")

def plot_ideality_vs_voltage(doping, srv, csv_path='results/sweep_results.csv', output_dir='plots'):
    """
    Plots ideality factor vs voltage for different tau_srh for a fixed doping and SRV.
    Calculates ideality factor from the saved dark IV data.
    """
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    # Load sweep data to find matching conditions
    df = pd.read_csv(csv_path)
    
    # Filter for the specific doping and SRV with some tolerance for floats
    mask = (np.isclose(df['doping'], doping, rtol=1e-5)) & (np.isclose(df['srv'], srv, rtol=1e-5))
    df_sub = df[mask]
    
    if df_sub.empty:
        print(f"No data found for doping={doping:.1e} and srv={srv:.1e} in {csv_path}")
        return

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tau_levels = sorted(df_sub['tau_srh'].unique())
    print(f"Found {len(tau_levels)} tau_srh levels for doping={doping:.1e}, srv={srv:.1e}")

    plt.figure(figsize=(8, 6))
    kBT = cfg.KB * cfg.T_DEFAULT
    
    for tau in tau_levels:
        # Construct filename consistent with GaAsCellCalculator.save_simulation_results
        base_name = f"doping_{doping:.1e}_srv_{srv:.1e}_tau_{tau:.1e}"
        iv_path = os.path.join('results', f"{base_name}_iv.csv")
        
        if os.path.exists(iv_path):
            iv_df = pd.read_csv(iv_path)
            v = iv_df['Voltage (V)'].values
            j_dark = iv_df['Dark Current (mA/cm2)'].values
            
            # Filter out non-positive currents for log calculation
            valid = j_dark > 0
            if np.sum(valid) < 5:
                print(f"Warning: Not enough valid dark current points for tau={tau:.1e}")
                continue
                
            v_valid = v[valid]
            j_valid = j_dark[valid]
            
            # Interpolate for smoother differentiation
            # Using cubic spline for better derivative estimation
            f_j = interp1d(v_valid, j_valid, kind='cubic', bounds_error=False, fill_value="extrapolate")
            
            # Define voltage range for plotting (typically where dark current is dominant)
            # We want to avoid extreme edges where extrapolation might be weird
            v_plot = np.linspace(max(0.3, v_valid.min()), min(1.1, v_valid.max()), 100)
            dv = 0.001
            
            n = []
            for vp in v_plot:
                j1, j0 = f_j(vp + dv), f_j(vp)
                if j1 <= 0 or j0 <= 0:
                    n.append(np.nan)
                else:
                    # n = (1/kBT) * (dlnJ/dV)^-1
                    # (ln(j1) - ln(j0)) / dv is approx dlnJ/dV
                    ideality = (kBT ** -1) * (((np.log(j1) - np.log(j0)) / dv) ** -1)
                    n.append(ideality)
            
            plt.plot(v_plot, n, label=f'$\\tau_{{SRH}}$ = {tau:.1e} s')
        else:
            print(f"Warning: IV file not found: {iv_path}")
    
    plt.xlabel('Voltage (V)', fontsize=12)
    plt.ylabel('Ideality Factor (n)', fontsize=12)
    plt.title(f'Ideality Factor vs Voltage\nDoping = {doping:.1e} $cm^{-3}$, SRV = {srv:.1e} cm/s', fontsize=14)
    plt.legend(title='Lifetime', loc='best', frameon=True)
    plt.grid(True, alpha=0.3)
    plt.ylim(0.8, 2.5) # Typical range for ideality factor
    
    safe_doping = f"{doping:.1e}".replace('+', '')
    safe_srv = f"{srv:.1e}".replace('+', '')
    filename = os.path.join(output_dir, f'ideality_vs_v_doping_{safe_doping}_srv_{safe_srv}.png')
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Generated ideality vs voltage plot: {filename}")

if __name__ == "__main__":
    # Generate contour plots for all doping levels
    plot_ideality_contours()
    
    # Example: Plot ideality vs voltage for a specific case
    # Let's pick a middle-of-the-road doping and SRV from the sweep
    example_doping = 8.0e19
    example_srv = 1.0e7
    plot_ideality_vs_voltage(example_doping, example_srv)
