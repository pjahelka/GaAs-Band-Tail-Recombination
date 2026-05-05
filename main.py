import numpy as np
from scipy.integrate import quad, trapezoid
from scipy.optimize import root_scalar, minimize_scalar
from scipy.interpolate import interp1d
import warnings

# Suppress integration warnings for cleaner output
warnings.filterwarnings("ignore")


class GaAsCellCalculator:
    def __init__(self, doping=6e19, srv=1E7, We=200.0e-7, Wd=200.0e-7, Lp=10.0e-4, ):
        # ---------------------------------------------------------
        # Physical Constants and Parameters
        # ---------------------------------------------------------
        self.c = 3e10  # speed of light, cm/s
        self.h = 4.14e-15  # Planck constant, eVs
        self.kB = 8.62e-5  # Boltzmann constant, eV/K
        self.T = 300
        self.kBT = self.kB * self.T
        self.q = 1.602e-16  # fundamental charge, mC

        # ---------------------------------------------------------
        # GaAs properties
        # ---------------------------------------------------------
        self.cbb = 1.3e-10  # GaAs radiative recombination coefficient
        self.ni = 2.0e6  # GaAs intrinsic carrier concentration
        self.nv = 9.0e18  # valence band density of states
        self.nc = 4.7e17  # conduction band density of states
        self.n = 3.5  # Band edge index of refraction
        self.eg = 1.42  # unperturbed bandgap
        self.alpha_bandgap = 1000.0
        self.srv = srv  # Electron surface recombination velocity

        # ---------------------------------------------------------
        # Device geometry
        # ---------------------------------------------------------
        self.We = We  # Emitter Width
        self.Wd = Wd  # Depletion Width
        self.Lp = Lp  # Base Width

        # ---------------------------------------------------------
        # Simulation settings
        # ---------------------------------------------------------
        self.dv = 0.005
        self.vmax = 1.2
        self.vmin = -0.1
        self.jl = -30  # Photocurrent
        self.psun = 100  # AM1.5G
        self.e0 = 0
        self.de = 0.005
        self.delta_mu0 = 0
        self.d_delta_mu = 0.005
        self.p_emit = doping

        # State variables
        self.eu = None
        self.eg_emit = None
        self.Dn = None
        self.efp = None
        self.np0 = None
        self.Ln = None
        self.alpha_factor = None

        # Interpolators
        self._i_func_interpf = None
        self._ecd_interpf = None
        self.cellj_light_interpf = None

        # Run initialization
        self.initialize(doping)

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

        val, _ = quad(integrand, -np.inf, 0, epsabs=1e-4, epsrel=1e-4)
        return self.nv * (2.0 / np.sqrt(np.pi)) * val

    def n_edge(self, v):
        return self.nc * np.exp(-(self.eg_emit - (self.efp + v)) / self.kBT)

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
        return self.kBT * np.log(np.maximum(self.n_emit(x, v), 1e-20) / self.nc) + self.eg_emit

    def j_surf(self, v):
        numerator = self.q * self.Dn * ((-self.np0 + self.n_edge(v)) * self.srv)
        denominator = self.Dn * np.cosh(self.We / self.Ln) + self.Ln * self.srv * np.sinh(self.We / self.Ln)
        return numerator / denominator

    def delta_mu(self, x, v):
        return self.efn(x, v) - self.efp

    def jdb_dark(self, v):
        x = v / self.kBT
        # term1 + term2 + term3 = Wd*(exp(x)-1) - Lp + Lp*(exp(x)-1)/x
        return self.q * self.cbb * (self.ni ** 2) * (self.Wd * np.expm1(x) - self.Lp + self.Lp * self._expm1_over_x(x))

    def jdb_light(self, v):
        return self.jl + self.jdb_dark(v)

    def alpha_integrand(self, ev, d_mu, e):
        """Original alpha integrand for reference."""
        if ev > 0: return 0.0
        return np.sqrt(-ev) * np.exp((ev + e) / self.eu) * (self.fd(ev, self.efp) - self.fd(ev + e, self.efp + d_mu))

    def alpha_base(self, e, d_mu):
        """Optimized alpha calculation using pre-computed I_func."""
        if np.isscalar(e) and np.isscalar(d_mu):
            if e >= self.eg_emit: return self.alpha_bandgap
            v_eff = self.efp + d_mu - e
            return self.alpha_factor * np.exp(e / self.eu) * (
                        self._i_func_interpf(self.efp) - self._i_func_interpf(v_eff))

        # Array version (for table generation)
        e_arr = np.atleast_1d(e)
        d_mu_arr = np.atleast_1d(d_mu)
        res = np.full((len(d_mu_arr), len(e_arr)), self.alpha_bandgap)
        mask = e_arr < self.eg_emit
        if np.any(mask):
            e_below = e_arr[mask]
            v_eff = self.efp + d_mu_arr[:, np.newaxis] - e_below[np.newaxis, :]
            i_vals = self._i_func_interpf(v_eff)
            i_ref = self._i_func_interpf(self.efp)
            res[:, mask] = self.alpha_factor * np.exp(e_below / self.eu) * (i_ref - i_vals)
        return res if res.shape != (1, 1) else res[0, 0]

    def alpha_interp(self, e, d_mu):
        return self.alpha_base(e, d_mu)

    def jbb(self, e, d_mu):
        factor = (8 * np.pi * self.n ** 2 * self.c) / ((self.h * self.c) ** 3)
        return factor * (e ** 2) * self.be(e, d_mu)

    def current_integrand(self, e, d_mu):
        return self.alpha_base(e, d_mu) * self.jbb(e, d_mu)

    def emitter_current_density(self, d_mu):
        return self._ecd_interpf(d_mu)

    def emitter_current(self, v):
        x_grid = np.linspace(0, self.We, 50)
        dmu_vals = self.delta_mu(x_grid, v)
        ecd_vals = self.emitter_current_density(dmu_vals)
        return trapezoid(ecd_vals, x_grid)

    def cellj_dark(self, v):
        return self.jdb_dark(v) + self.emitter_current(v) + self.j_surf(v)

    def cellj_light(self, v):
        return self.cellj_dark(v) + self.jl

    def cellj_light_interp(self, v):
        return self.cellj_light_interpf(v)

    def calc_pmax(self, j_func):
        res = minimize_scalar(lambda x: x * j_func(x), bounds=(self.vmin, self.vmax), method='bounded')
        return -res.fun, res.x

    def calc_jsc(self, j_func):
        return j_func(0)

    def calc_voc(self, j_func):
        try:
            v1, v2 = 0.5 * (self.vmin + self.vmax), self.vmax
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
        return {"pmax": pmax, "vmpp": vmpp, "jmpp": jmpp, "eff": pmax / self.psun, "jsc": jsc, "voc": voc, "ff": ff}

    def initialize(self):
        print(f"Initializing parameters for doping = {self.doping:e}...")
        self.Dn = 35 + 28 * np.log(self.p_emit / 1e19)
        self.eu = 1.0e-3 * (12 + 6 * self.p_emit / 1e19)
        self.eg_emit = self.eg - 1.6e-8 * (self.p_emit ** (1 / 3))
        emax = self.eg_emit + 0.3

        res_efp = root_scalar(lambda x: self.p_func(x) - self.p_emit, bracket=[-0.5, 0.5])  # Find the hole QFL
        self.efp = res_efp.root
        self.np0 = self.n_edge(0)
        self.Ln = 1.7e-4 * ((self.p_emit / 1e19) ** -0.4)

        # Precompute Alpha Integral helper function
        def i_func_raw(v):
            return quad(lambda ev: np.sqrt(-ev) * np.exp(ev / self.eu) * self.fd(ev, v), -40 * self.eu, 0)[0]

        print("Pre-computing Alpha and Emitter Current Density tables...")
        v_vals = np.linspace(self.efp - emax - 0.1, self.efp + emax + 0.1, 200)
        self._i_func_interpf = interp1d(v_vals, [i_func_raw(v) for v in v_vals], kind='cubic', fill_value='extrapolate')

        i_ref = self._i_func_interpf(self.efp)
        val_alpha = np.exp(self.eg_emit / self.eu) * (i_ref - self._i_func_interpf(self.efp - self.eg_emit))
        self.alpha_factor = self.alpha_bandgap / val_alpha

        # Emitter Current Density Table (Ndmu, Ne)
        dmu_grid = np.linspace(0, emax, 100)
        e_grid = np.linspace(0, emax, 250)
        alpha_tab = self.alpha_base(e_grid, dmu_grid)
        jbb_tab = self.jbb(e_grid[np.newaxis, :], dmu_grid[:, np.newaxis])
        ecd_vals = self.q * trapezoid(alpha_tab * jbb_tab, e_grid, axis=1)
        self._ecd_interpf = interp1d(dmu_grid, ecd_vals, kind='cubic', fill_value='extrapolate')

        # Cell Light Current Table
        print("Pre-computing light current density grid...")
        v_grid = np.arange(self.vmin, self.vmax + self.dv, self.dv)
        self.cellj_light_interpf = interp1d(v_grid, [self.cellj_light(v) for v in v_grid],
                                            kind='linear', bounds_error=False, fill_value="extrapolate")
        print("Initialization complete.\n")


if __name__ == "__main__":
    cell = GaAsCellCalculator(doping=8e19)
    pv = cell.calc_pv(cell.cellj_light_interp)
    print("Photovoltaic Parameters:")
    for k, v in pv.items(): print(f"  {k}: {v:.6f}")
    print(f"\nIdeality Factor: {cell.calc_ideality(cell.cellj_dark, pv['vmpp']):.6f}")
