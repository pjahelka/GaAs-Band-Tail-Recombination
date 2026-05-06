import numpy as np
from scipy.integrate import quad, trapezoid
from scipy.optimize import root_scalar, minimize_scalar
from scipy.interpolate import interp1d, RectBivariateSpline
import warnings
import multiprocessing
import os
import csv
import config as cfg

# Suppress integration warnings for cleaner output
warnings.filterwarnings("ignore")


# Helper functions for parallelization
def fd_func(e, mu, kBT):
    val = np.clip((e - mu) / kBT, -700, 700)
    return 1.0 / (np.exp(val) + 1.0)


def be_func(e, mu, kBT):
    val = np.clip((e - mu) / kBT, -700, 700)
    exp_val = np.exp(val)
    return 1.0 / np.where(exp_val == 1.0, 1e-10, exp_val - 1.0)


def alpha_integrand_func(ev, d_mu, e, eu, efp, kBT):
    if ev > 0: return 0.0
    return np.sqrt(-ev) * np.exp((ev + e) / eu) * (fd_func(ev, efp, kBT) - fd_func(ev + e, efp + d_mu, kBT))


def alpha_raw_worker(args):
    e, d_mu, eg_emit, alpha_bandgap, alpha_factor, eu, efp, kBT, e_min = args
    if e >= eg_emit:
        return alpha_bandgap
    # Using e_min as a safe lower limit for the Urbach tail integration
    val, _ = quad(alpha_integrand_func, e_min, 0, args=(d_mu, e, eu, efp, kBT), epsabs=cfg.EPSABS, epsrel=cfg.EPSREL)
    return alpha_factor * val


def ecd_worker(args):
    dmu, emax, eg_emit, q, alpha_interpf, n, c, h, kBT, alpha_bandgap = args

    def local_integrand(e):
        if e >= eg_emit:
            alpha = alpha_bandgap
        else:
            alpha = alpha_interpf(e, dmu)[0, 0]

        factor = (8 * np.pi * n ** 2 * c) / ((h * c) ** 3)
        jbb_val = factor * (e ** 2) * be_func(e, dmu, kBT)
        return alpha * jbb_val

    val_ecd, _ = quad(local_integrand, 0, emax, points=[eg_emit], epsabs=cfg.EPSABS, epsrel=cfg.EPSREL)
    return q * val_ecd


class GaAsCellCalculator:
    def __init__(self, doping=cfg.DOPING_DEFAULT, srv=cfg.SRV_DEFAULT, 
                 We=cfg.WE_DEFAULT, Wd=cfg.WD_DEFAULT, Lp=cfg.LP_DEFAULT, tau_SRH = cfg.TAU_SRH):
        # ---------------------------------------------------------
        # Physical Constants and Parameters
        # ---------------------------------------------------------
        self.kBT = cfg.KB * cfg.T_DEFAULT

        # ---------------------------------------------------------
        # GaAs properties
        # ---------------------------------------------------------
        self.srv = srv

        # ---------------------------------------------------------
        # Device geometry
        # ---------------------------------------------------------
        self.We = We
        self.Wd = Wd
        self.Lp = Lp
        self.TAU_SRH = tau_SRH
        # ---------------------------------------------------------
        # Simulation settings
        # ---------------------------------------------------------
        self.p_emit = doping
        self.doping = doping

        # State variables
        self.eu = None
        self.eg_emit = None
        self.Dn = None
        self.efp = None
        self.np0 = None
        self.Ln = None
        self.alpha_factor = None

        # Interpolators
        self._alpha_interpf = None
        self._ecd_interpf = None
        self.cellj_light_interpf = None

        # Run initialization
        self.initialize()

    def fd(self, e, mu):
        val = np.clip((e - mu) / self.kBT, -700, 700)
        return 1.0 / (np.exp(val) + 1.0)

    def be(self, e, mu):
        val = np.clip((e - mu) / self.kBT, -700, 700)
        exp_val = np.exp(val)
        return 1.0 / np.where(exp_val == 1.0, 1e-10, exp_val - 1.0)

    def _expm1_over_x(self, x):
        """Stable calculation of (exp(x)-1)/x"""
        return np.where(np.abs(x) < 1e-7, 1.0 + x / 2.0, np.expm1(x) / x)

    def p_func(self, efp):
        def integrand(e):
            return np.sqrt(-e / self.kBT) * (1.0 - self.fd(e, efp)) / self.kBT

        val, _ = quad(integrand, cfg.E_MIN, 0, epsabs=cfg.EPSABS, epsrel=cfg.EPSREL)
        return cfg.NV * (2.0 / np.sqrt(np.pi)) * val

    def n_edge(self, v):
        return cfg.NC * np.exp(-(self.eg_emit - (self.efp + v)) / self.kBT)

    def n_emit(self, x, v):
        # Optimized for scalar or array x
        term1 = 2 * np.exp(self.We / self.Ln)
        term2 = self.Dn * self.np0 * np.cosh(self.We / self.Ln) + self.Ln * self.np0 * self.srv * np.sinh(
            self.We / self.Ln)
        term3 = (self.np0 - self.n_edge(v)) * (
                    self.Dn * np.cosh(x / self.Ln) + self.Ln * self.srv * np.sinh(x / self.Ln))
        denominator = self.Dn - self.Ln * self.srv + np.exp((2 * self.We) / self.Ln) * (self.Dn + self.Ln * self.srv)
        return term1 * (term2 - term3) / denominator

    def efn(self, x, v):
        return self.kBT * np.log(np.maximum(self.n_emit(x, v), 1e-20) / cfg.NC) + self.eg_emit

    def j_surf(self, v):
        numerator = cfg.Q * self.Dn * ((-self.np0 + self.n_edge(v)) * self.srv)
        denominator = self.Dn * np.cosh(self.We / self.Ln) + self.Ln * self.srv * np.sinh(self.We / self.Ln)
        return numerator / denominator

    def j_srh(self, v):
        return cfg.Q * cfg.NI * (np.exp(v / (2 * self.kBT)) - 1) * self.Wd * np.sqrt(1 - v / cfg.VBI) / (2 * self.TAU_SRH)

    def delta_mu(self, x, v):
        return self.efn(x, v) - self.efp

    def jdb_dark(self, v):
        x = v / self.kBT
        # term1 + term2 + term3 = Wd*(exp(x)-1) - Lp + Lp*(exp(x)-1)/x
        return cfg.Q * cfg.CBB * (cfg.NI ** 2) * (self.Wd * np.expm1(x) - self.Lp + self.Lp * self._expm1_over_x(x))

    def alpha_integrand(self, ev, d_mu, e):
        """Alpha integrand exactly as specified in the text file."""
        if ev > 0: return 0.0
        return np.sqrt(-ev) * np.exp((ev + e) / self.eu) * (self.fd(ev, self.efp) - self.fd(ev + e, self.efp + d_mu))

    def alpha_raw(self, e, d_mu):
        """Direct calculation of alpha using numerical integration."""
        if e >= self.eg_emit:
            return cfg.ALPHA_BANDGAP
        val, _ = quad(self.alpha_integrand, cfg.E_MIN, 0, args=(d_mu, e), epsabs=cfg.EPSABS, epsrel=cfg.EPSREL)
        return self.alpha_factor * val

    def alpha_interp(self, e, d_mu):
        """2D Interpolated alpha calculation."""
        if np.isscalar(e) and np.isscalar(d_mu):
            return cfg.ALPHA_BANDGAP if e >= self.eg_emit else self._alpha_interpf(e, d_mu)[0, 0]
        
        # Simplified vectorized grid evaluation
        e_arr, dmu_arr = np.atleast_1d(e), np.atleast_1d(d_mu)
        res = self._alpha_interpf(e_arr, dmu_arr).T
        res[:, e_arr >= self.eg_emit] = cfg.ALPHA_BANDGAP
        return res.squeeze()

    def jbb(self, e, d_mu):
        factor = (8 * np.pi * cfg.N_INDEX ** 2 * cfg.C) / ((cfg.H * cfg.C) ** 3)
        return factor * (e ** 2) * self.be(e, d_mu)

    def current_integrand(self, e, d_mu):
        return self.alpha_interp(e, d_mu) * self.jbb(e, d_mu)

    def emitter_current_density(self, d_mu):
        """Returns the pre-computed emitter current density for a given delta_mu."""
        return self._ecd_interpf(d_mu)

    def emitter_current(self, v):
        x_grid = np.linspace(0, self.We, cfg.X_GRID_POINTS)
        dmu_vals = self.delta_mu(x_grid, v)
        ecd_vals = self.emitter_current_density(dmu_vals)
        return trapezoid(ecd_vals, x_grid)

    def cellj_dark(self, v):
        return self.jdb_dark(v) + self.emitter_current(v) + self.j_surf(v) + self.j_srh(v)

    def cellj_light(self, v):
        return self.cellj_dark(v) + cfg.JL_DEFAULT

    def cellj_light_interp(self, v):
        return self.cellj_light_interpf(v)

    def calc_pmax(self, j_func):
        res = minimize_scalar(lambda x: x * j_func(x), bounds=(cfg.VMIN, cfg.VMAX), method='bounded')
        return -res.fun, res.x

    def calc_jsc(self, j_func):
        return j_func(0)

    def calc_voc(self, j_func):
        try:
            v1, v2 = 0.5 * (cfg.VMIN + cfg.VMAX), cfg.VMAX
            if j_func(v1) * j_func(v2) > 0: v1 = 0
            res = root_scalar(j_func, bracket=[v1, v2])
            return res.root
        except ValueError:
            return np.nan

    def calc_ideality(self, j_func, v):
        dv = 0.001
        j1, j0 = j_func(v + dv), j_func(v)
        if j1 <= 0 or j0 <= 0: return np.nan
        return (self.kBT ** -1) * (((np.log(j1) - np.log(j0)) / dv) ** -1)

    def calc_pv(self, j_func):
        pmax, vmpp = self.calc_pmax(j_func)
        jmpp = j_func(vmpp)
        jsc, voc = -self.calc_jsc(j_func), self.calc_voc(j_func)
        ff = pmax / (jsc * voc) if (jsc * voc) != 0 else np.nan
        return {"pmax": pmax, "vmpp": vmpp, "jmpp": jmpp, "eff": pmax / cfg.PSUN, "jsc": jsc, "voc": voc, "ff": ff}

    def save_simulation_results(self, results_dir="results", verbose=True):
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        pv = self.calc_pv(self.cellj_light_interp)
        ideality = self.calc_ideality(self.cellj_dark, pv['vmpp'])
        pv['ideality'] = ideality

        if verbose:
            print("Photovoltaic Parameters:")
            for k, v in pv.items(): print(f"  {k}: {v:.6f}")

            # Calculate and print Efn at the surface (x=0) at Voc
            voc = pv['voc']
            efn_surface = self.efn(0, voc)
            print(f"\nEfn at surface (x=0) at Voc: {efn_surface:.6f} eV")
            print(f"Ideality Factor: {ideality:.6f}")

        # File naming components
        base_name = f"doping_{self.doping:.1e}_srv_{self.srv:.1e}_tau_{self.TAU_SRH:.1e}"

        # 1. Save Headline PV results
        pv_filename = os.path.join(results_dir, f"{base_name}_pv.csv")
        with open(pv_filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(pv.keys())
            writer.writerow(pv.values())
        if verbose:
            print(f"PV results saved to {pv_filename}")

        # 2. Save Light and Dark IVs
        iv_filename = os.path.join(results_dir, f"{base_name}_iv.csv")
        v_grid = np.linspace(cfg.VMIN, cfg.VMAX, 200) # Use a reasonably dense grid for IV
        dark_currents = [self.cellj_dark(v) for v in v_grid]
        light_currents = [self.cellj_light_interp(v) for v in v_grid]

        with open(iv_filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Voltage (V)", "Dark Current (mA/cm2)", "Light Current (mA/cm2)"])
            for v, dark, light in zip(v_grid, dark_currents, light_currents):
                writer.writerow([v, dark, light])
        if verbose:
            print(f"IV results saved to {iv_filename}")

        return pv

    def initialize(self):
        print(f"Initializing parameters for doping = {self.p_emit:e}...")
        self.Dn = cfg.DN_BASE + cfg.DN_COEFF * np.log(self.p_emit / cfg.REF_DOPING)
        self.eu = 1.0e-3 * (cfg.EU_BASE + cfg.EU_COEFF * self.p_emit / cfg.REF_DOPING)
        self.eg_emit = cfg.EG_UNPERTURBED - cfg.EG_EMIT_COEFF * (self.p_emit ** (1 / 3))
        self.emax = self.eg_emit + cfg.EMAX_OFFSET

        res_efp = root_scalar(lambda x: self.p_func(x) - self.p_emit, bracket=cfg.EFP_BRACKET)  # Find the hole QFL
        self.efp = res_efp.root
        self.np0 = self.n_edge(0)
        self.Ln = cfg.LN_BASE * ((self.p_emit / cfg.REF_DOPING) ** cfg.LN_COEFF)

        # Calculate alpha_factor
        print("Calculating alpha scaling factor...")
        val_init, _ = quad(lambda ev: self.alpha_integrand(ev, 0, self.eg_emit), cfg.E_MIN, 0, epsabs=cfg.EPSABS, epsrel=cfg.EPSREL)
        self.alpha_factor = cfg.ALPHA_BANDGAP / val_init

        # Pre-compute alpha interpolation table
        print("Pre-computing alpha(e, delta_mu) interpolation table...")
        e_grid = np.linspace(cfg.E0, self.emax, int((self.emax - cfg.E0)/cfg.DE))
        dmu_grid = np.linspace(cfg.DELTA_MU0, cfg.VMAX, int((cfg.VMAX - cfg.DELTA_MU0)/cfg.D_DELTA_MU))
        
        args_list = []
        for e_val in e_grid:
            for dmu_val in dmu_grid:
                args_list.append((e_val, dmu_val, self.eg_emit, cfg.ALPHA_BANDGAP, self.alpha_factor, self.eu, self.efp, self.kBT, cfg.E_MIN))
        
        with multiprocessing.Pool() as pool:
            results = pool.map(alpha_raw_worker, args_list)
        
        alpha_values = np.array(results).reshape(len(e_grid), len(dmu_grid))
        self._alpha_interpf = RectBivariateSpline(e_grid, dmu_grid, alpha_values)

        # Pre-compute Emitter Current Density interpolation table
        print("Pre-computing emitter current density table...")
        # ECD depends only on delta_mu
        dmu_ecd_grid = np.linspace(cfg.VMIN - cfg.VMIN_OFFSET, self.emax, cfg.DMU_ECD_GRID_POINTS)
        
        args_list_ecd = []
        for dmu in dmu_ecd_grid:
            args_list_ecd.append((dmu, self.emax, self.eg_emit, cfg.Q, self._alpha_interpf, cfg.N_INDEX, cfg.C, cfg.H, self.kBT, cfg.ALPHA_BANDGAP))
        
        with multiprocessing.Pool() as pool:
            ecd_values = pool.map(ecd_worker, args_list_ecd)
            
        self._ecd_interpf = interp1d(dmu_ecd_grid, ecd_values, kind='cubic', bounds_error=False, fill_value="extrapolate")

        # Cell Light Current Table
        print("Pre-computing light current density grid...")
        v_grid = np.arange(cfg.VMIN, cfg.VMAX + cfg.DV, cfg.DV)
        self.cellj_light_interpf = interp1d(v_grid, [self.cellj_light(v) for v in v_grid],
                                            kind='linear', bounds_error=False, fill_value="extrapolate")
        print("Initialization complete.\n")


if __name__ == "__main__":
    # Simulation parameters
    doping = 2e19
    srv = 1E0
    tau_SRH = 1E-10

    cell = GaAsCellCalculator(doping=doping, srv=srv, tau_SRH=tau_SRH)
    cell.save_simulation_results()
