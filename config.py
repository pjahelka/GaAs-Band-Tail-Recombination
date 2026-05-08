# Physical Constants
C = 3e10            # speed of light, cm/s
H = 4.14e-15        # Planck constant, eVs
KB = 8.62e-5        # Boltzmann constant, eV/K
Q = 1.602e-16       # fundamental charge, mC
T_DEFAULT = 300     # Default temperature, K

# GaAs Material Constants
CBB = 1.3e-10       # GaAs radiative recombination coefficient
NI = 2.0e6          # GaAs intrinsic carrier concentration
NV = 9.0e18         # valence band density of states
NC = 4.7e17         # conduction band density of states
N_INDEX = 3.5       # Band edge index of refraction
EG_UNPERTURBED = 1.42  # unperturbed bandgap, eV
ALPHA_BANDGAP = 1000.0 # Absorption at bandgap, cm^-1

# Empirical GaAs parameters (formula coefficients)
DN_BASE = 35.0
DN_COEFF = 28.0
EU_BASE = 12.0
EU_COEFF = 6.0
EG_EMIT_COEFF = 1.6e-8
LN_BASE = 1.7e-4
LN_COEFF = -0.4
REF_DOPING = 1e19
TAU_SRH = 1e-6 #SRH lifetime
VBI = 1.42

# Device Default Geometry & Properties
WE_DEFAULT = 200.0e-7  # Emitter Width, cm
WD_DEFAULT = 200.0e-7  # Depletion Width, cm
LP_DEFAULT = 10.0e-4   # Base Width, cm
DOPING_DEFAULT = 6e19  # p-type doping, cm^-3
SRV_DEFAULT = 1E7      # Surface recombination velocity, cm/s

# Simulation Parameters
RESULTS_DIR = "results"
PLOTS_DIR = "plots"
DV = 0.005             # Voltage step for grid, V
VMAX = 1.2             # Maximum voltage, V
VMIN = -0.1            # Minimum voltage, V
JL_DEFAULT = -30       # Photocurrent density, mA/cm^2
PSUN = 100             # Solar power density (AM1.5G), mW/cm^2
E_MIN = -5             # Minimum energy for integration, eV
E0 = 0                 # Base energy for integration, eV
DE = 0.005             # Energy step for pre-computation, eV
DELTA_MU0 = 0          # Initial delta_mu for pre-computation
D_DELTA_MU = 0.005     # delta_mu step for pre-computation

# Numeric Parameters (Integration & Interpolation)
EPSABS = 1e-4
EPSREL = 1e-4
X_GRID_POINTS = 50
DMU_ECD_GRID_POINTS = 100
EMAX_OFFSET = 0.3      # Offset for maximum energy in grids
VMIN_OFFSET = 0.1      # Offset for minimum voltage in grids
EFP_BRACKET = [-0.5, 0.5] # Bracket for hole QFL root finding
