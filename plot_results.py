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

def plot_ideality_contours(csv_path=None, output_dir=None):
    """
    Reads sweep results and generates contour plots of ideality factor vs SRV and Tau_SRH
    for each unique doping level.
    """
    if csv_path is None:
        csv_path = os.path.join(cfg.RESULTS_DIR, 'sweep_results.csv')
    if output_dir is None:
        output_dir = cfg.PLOTS_DIR

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
        
        cp = plt.contourf(X, Y, Z, levels=levels, cmap='viridis')
        cbar = plt.colorbar(cp)
        cbar.set_label('Ideality Factor (n)', fontsize=12)
        
        # Set scales to log as SRV and Tau usually span orders of magnitude
        plt.xscale('log')
        plt.yscale('log')
        
        plt.xlabel('Surface Recombination Velocity (SRV) [cm/s]', fontsize=12)
        plt.ylabel('SRH Lifetime ($\\tau_{SRH}$) [s]', fontsize=12)
        plt.title(f'Ideality Factor Contour\nDoping = {doping:.1e} $cm^{{-3}}$', fontsize=14)
        
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

def plot_ideality_vs_voltage(doping, srv, csv_path=None, results_dir=None, output_dir=None):
    """
    Plots ideality factor vs voltage for different tau_srh for a fixed doping and SRV.
    Calculates ideality factor from the saved dark IV data.
    """
    if csv_path is None:
        if results_dir is None:
            csv_path = os.path.join(cfg.RESULTS_DIR, 'sweep_results.csv')
        else:
            csv_path = os.path.join(results_dir, 'sweep_results.csv')
    
    if results_dir is None:
        results_dir = cfg.RESULTS_DIR
        
    if output_dir is None:
        output_dir = cfg.PLOTS_DIR

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
        iv_path = os.path.join(results_dir, f"{base_name}_iv.csv")
        
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
            dv = 0.0001
            
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
    plt.title(f'Ideality Factor vs Voltage\nDoping = {doping:.1e} $cm^{{-3}}$, SRV = {srv:.1e} cm/s', fontsize=14)
    plt.legend(title='Lifetime', loc='best', frameon=True)
    plt.grid(True, alpha=0.3)
    plt.ylim(0.8, 2.5) # Typical range for ideality factor
    
    safe_doping = f"{doping:.1e}".replace('+', '')
    safe_srv = f"{srv:.1e}".replace('+', '')
    filename = os.path.join(output_dir, f'ideality_vs_v_doping_{safe_doping}_srv_{safe_srv}.png')
    plt.savefig(filename, dpi=300)
    #plt.show()
    print(f"Generated ideality vs voltage plot: {filename}")

def _generate_contour_plots_by_srv(df, value_col, output_dir, title_prefix, label, filename_suffix, cmap='viridis', multiplier=1.0):
    """
    Helper function to generate contour plots for a given value column, 
    grouped by SRV, with Doping on X and Tau_SRH on Y.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Apply multiplier (e.g., for V to mV conversion)
    if multiplier != 1.0:
        plot_val_col = f"{value_col}_scaled"
        df = df.copy()
        df[plot_val_col] = df[value_col] * multiplier
    else:
        plot_val_col = value_col

    srv_levels = sorted(df['srv'].unique())
    
    # Determine common color limits for consistency across all SRV plots
    v_min, v_max = df[plot_val_col].min(), df[plot_val_col].max()
    if np.isclose(v_min, v_max):
        levels = 21
    else:
        levels = np.linspace(v_min, v_max, 21)

    for srv in srv_levels:
        df_sub = df[df['srv'] == srv]
        
        try:
            pivot_df = df_sub.pivot(index='tau_srh', columns='doping', values=plot_val_col)
        except Exception as e:
            print(f"Error pivoting data for SRV {srv:.1e}: {e}")
            continue
            
        X = pivot_df.columns.values
        Y = pivot_df.index.values
        Z = pivot_df.values
        
        plt.figure(figsize=(10, 8))
        
        cp = plt.contourf(X, Y, Z, levels=levels, cmap=cmap)
        cbar = plt.colorbar(cp)
        cbar.set_label(label, fontsize=12)
        
        plt.xscale('log')
        plt.yscale('log')
        
        plt.xlabel(r'Doping ($cm^{-3}$)', fontsize=12)
        plt.ylabel(r'SRH Lifetime ($\tau_{SRH}$) [s]', fontsize=12)
        plt.title(f'{title_prefix}\nSRV = {srv:.1e} cm/s', fontsize=14)
        
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.tight_layout()
        
        safe_srv_str = f"{srv:.1e}".replace('+', '')
        filename = os.path.join(output_dir, f'{filename_suffix}_srv_{safe_srv_str}.png')
        plt.savefig(filename, dpi=300)
        plt.close()
    
    print(f"Generated plots in: {output_dir}")

def plot_voc_contours(results_dir='results', results_no_bgn_dir='results_no_bgn', output_dir=None):
    """
    Generates contour plots for Voc: with BGN, without BGN, and the difference.
    """
    if output_dir is None:
        output_dir = os.path.join(cfg.PLOTS_DIR, 'voc_comparison')

    csv_with = os.path.join(results_dir, 'sweep_results.csv')
    csv_without = os.path.join(results_no_bgn_dir, 'sweep_results.csv')
    
    if not os.path.exists(csv_with) or not os.path.exists(csv_without):
        print(f"Error: Required results files not found.")
        return

    df_with = pd.read_csv(csv_with)
    df_without = pd.read_csv(csv_without)
    
    # 1. Voc with BGN
    _generate_contour_plots_by_srv(
        df_with, 'voc', os.path.join(output_dir, 'voc_with_bgn'),
        'Voc (With BGN)', r'$V_{oc}$ (mV)', 'voc_with_bgn', multiplier=1000
    )
    
    # 2. Voc without BGN
    _generate_contour_plots_by_srv(
        df_without, 'voc', os.path.join(output_dir, 'voc_no_bgn'),
        'Voc (No BGN)', r'$V_{oc}$ (mV)', 'voc_no_bgn', multiplier=1000
    )
    
    # 3. Voc Difference
    merged = pd.merge(df_with, df_without, on=['doping', 'srv', 'tau_srh'], suffixes=('_with', '_without'))
    if not merged.empty:
        merged['voc_diff'] = merged['voc_with'] - merged['voc_without']
        _generate_contour_plots_by_srv(
            merged, 'voc_diff', os.path.join(output_dir, 'voc_diff'),
            'Voc Difference (With BGN - No BGN)', r'$\Delta V_{oc}$ (mV)', 'voc_diff', multiplier=1000
        )

def plot_ff_contours(results_dir='results', results_no_bgn_dir='results_no_bgn', output_dir=None):
    """
    Generates contour plots for FF: with BGN, without BGN, and the difference.
    """
    if output_dir is None:
        output_dir = os.path.join(cfg.PLOTS_DIR, 'ff_comparison')

    csv_with = os.path.join(results_dir, 'sweep_results.csv')
    csv_without = os.path.join(results_no_bgn_dir, 'sweep_results.csv')
    
    if not os.path.exists(csv_with) or not os.path.exists(csv_without):
        print(f"Error: Required results files not found.")
        return

    df_with = pd.read_csv(csv_with)
    df_without = pd.read_csv(csv_without)
    
    # 1. FF with BGN
    _generate_contour_plots_by_srv(
        df_with, 'ff', os.path.join(output_dir, 'ff_with_bgn'),
        'Fill Factor (With BGN)', 'FF', 'ff_with_bgn'
    )
    
    # 2. FF without BGN
    _generate_contour_plots_by_srv(
        df_without, 'ff', os.path.join(output_dir, 'ff_no_bgn'),
        'Fill Factor (No BGN)', 'FF', 'ff_no_bgn'
    )
    
    # 3. FF Difference
    merged = pd.merge(df_with, df_without, on=['doping', 'srv', 'tau_srh'], suffixes=('_with', '_without'))
    if not merged.empty:
        merged['ff_diff'] = merged['ff_with'] - merged['ff_without']
        _generate_contour_plots_by_srv(
            merged, 'ff_diff', os.path.join(output_dir, 'ff_diff'),
            'FF Difference (With BGN - No BGN)', r'$\Delta FF$', 'ff_diff'
        )

def plot_voc_diff_contours(results_dir='results', results_no_bgn_dir='results_no_bgn', output_dir=None):
    """
    Generates contour plots of Voc difference (with BGN - without BGN) 
    vs Doping and Tau_SRH for each unique SRV.
    """
    if output_dir is None:
        output_dir = os.path.join(cfg.PLOTS_DIR, 'voc_diff')
        
    csv_with = os.path.join(results_dir, 'sweep_results.csv')
    csv_without = os.path.join(results_no_bgn_dir, 'sweep_results.csv')
    
    if not os.path.exists(csv_with) or not os.path.exists(csv_without):
        print(f"Error: Required results files not found.")
        return

    df_with = pd.read_csv(csv_with)
    df_without = pd.read_csv(csv_without)
    
    merged = pd.merge(df_with, df_without, on=['doping', 'srv', 'tau_srh'], suffixes=('_with', '_without'))
    
    if merged.empty:
        print("Error: No matching simulation conditions found.")
        return

    merged['voc_diff'] = merged['voc_with'] - merged['voc_without']
    
    _generate_contour_plots_by_srv(
        merged, 'voc_diff', output_dir,
        'Voc Difference (With BGN - No BGN)', r'$\Delta V_{oc}$ (mV)', 'voc_diff', multiplier=1000
    )

def plot_voc_vs_doping(srv, csv_path=None, output_dir=None):
    """
    Plots Voc vs Doping for different tau_srh at a fixed SRV.
    """
    if csv_path is None:
        csv_path = os.path.join(cfg.RESULTS_DIR, 'sweep_results.csv')
    if output_dir is None:
        output_dir = cfg.PLOTS_DIR

    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    df_sub = df[np.isclose(df['srv'], srv, rtol=1e-5)]

    if df_sub.empty:
        print(f"No data found for SRV = {srv:.1e}")
        return

    plt.figure(figsize=(8, 6))
    tau_levels = sorted(df_sub['tau_srh'].unique())

    for tau in tau_levels:
        df_tau = df_sub[df_sub['tau_srh'] == tau].sort_values('doping')
        plt.plot(df_tau['doping'], df_tau['voc'] * 1000, label=f'$\\tau_{{SRH}}$ = {tau:.1e} s')

    # plt.xscale('log')
    plt.xlabel(r'Doping ($cm^{-3}$)', fontsize=12)
    plt.ylabel(r'$V_{oc}$ (mV)', fontsize=12)
    plt.title(f'$V_{{oc}}$ vs Doping\nSRV = {srv:.1e} cm/s', fontsize=14)
    plt.legend(title='Lifetime', loc='best', frameon=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    safe_srv = f"{srv:.1e}".replace('+', '')
    filename = os.path.join(output_dir, f'voc_vs_doping_srv_{safe_srv}.png')
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Generated: {filename}")

def plot_efficiency_vs_doping(srv, csv_path=None, output_dir=None):
    """
    Plots Efficiency vs Doping for different tau_srh at a fixed SRV.
    """
    if csv_path is None:
        csv_path = os.path.join(cfg.RESULTS_DIR, 'sweep_results.csv')
    if output_dir is None:
        output_dir = cfg.PLOTS_DIR

    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    df_sub = df[np.isclose(df['srv'], srv, rtol=1e-5)]

    if df_sub.empty:
        print(f"No data found for SRV = {srv:.1e}")
        return

    plt.figure(figsize=(8, 6))
    tau_levels = sorted(df_sub['tau_srh'].unique())

    for tau in tau_levels:
        df_tau = df_sub[df_sub['tau_srh'] == tau].sort_values('doping')
        plt.plot(df_tau['doping'], df_tau['eff'] * 100, label=f'$\\tau_{{SRH}}$ = {tau:.1e} s')

    # plt.xscale('log')
    plt.xlabel(r'Doping ($cm^{-3}$)', fontsize=12)
    plt.ylabel('Efficiency (%)', fontsize=12)
    plt.title(f'Efficiency vs Doping\nSRV = {srv:.1e} cm/s', fontsize=14)
    plt.legend(title='Lifetime', loc='best', frameon=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    safe_srv = f"{srv:.1e}".replace('+', '')
    filename = os.path.join(output_dir, f'efficiency_vs_doping_srv_{safe_srv}.png')
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Generated: {filename}")

def plot_ideality_at_mpp_vs_doping(srv, csv_path=None, output_dir=None):
    """
    Plots Ideality Factor at MPP vs Doping for different tau_srh at a fixed SRV.
    """
    if csv_path is None:
        csv_path = os.path.join(cfg.RESULTS_DIR, 'sweep_results.csv')
    if output_dir is None:
        output_dir = cfg.PLOTS_DIR

    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    df_sub = df[np.isclose(df['srv'], srv, rtol=1e-5)]

    if df_sub.empty:
        print(f"No data found for SRV = {srv:.1e}")
        return

    plt.figure(figsize=(8, 6))
    tau_levels = sorted(df_sub['tau_srh'].unique())

    for tau in tau_levels:
        df_tau = df_sub[df_sub['tau_srh'] == tau].sort_values('doping')
        plt.plot(df_tau['doping'], df_tau['ideality'], label=f'$\\tau_{{SRH}}$ = {tau:.1e} s')

    # plt.xscale('log')
    plt.xlabel(r'Doping ($cm^{-3}$)', fontsize=12)
    plt.ylabel('Ideality Factor (n) at MPP', fontsize=12)
    plt.title(f'Ideality Factor vs Doping at MPP\nSRV = {srv:.1e} cm/s', fontsize=14)
    plt.legend(title='Lifetime', loc='best', frameon=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    safe_srv = f"{srv:.1e}".replace('+', '')
    filename = os.path.join(output_dir, f'ideality_at_mpp_vs_doping_srv_{safe_srv}.png')
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Generated: {filename}")

if __name__ == "__main__":
    # Generate Voc comparison contours (With BGN, No BGN, Difference)
    #plot_voc_contours()
    
    # Generate FF comparison contours (With BGN, No BGN, Difference)
    #plot_ff_contours()
    
    # The individual diff function can still be called
    # plot_voc_diff_contours()
    
    # Generate contour plots for all doping levels
    #plot_ideality_contours()
    
    # Example: Plot ideality vs voltage for a specific case
    # Let's pick a middle-of-the-road doping and SRV from the sweep
    example_doping = 5.0e19
    example_srv = 1.0e08
    # plot_ideality_vs_voltage(example_doping, example_srv)

    # Generate new plots requested
    plot_voc_vs_doping(example_srv)
    plot_efficiency_vs_doping(example_srv)
    plot_ideality_at_mpp_vs_doping(example_srv)
