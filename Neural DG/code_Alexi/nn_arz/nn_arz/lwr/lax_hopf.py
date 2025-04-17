import torch
import numpy as np

def force_float64(func):
    def wrapper(*args, **kwargs):
        # Convert all tensor arguments to float64
        args = [arg.to(torch.float64) if isinstance(arg, torch.Tensor) else arg for arg in args]
        kwargs = {k: v.to(torch.float64) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
        return func(*args, **kwargs)
    return wrapper

def greenshield_q(k, flux):
    return flux['v_max'] * k * (1.0 - k / flux['k_max'])

def greenshield_qp(k, flux):
    return flux['v_max'] * (1.0 - 2.0 * k / flux['k_max'])

def greenshield_R(u, flux):
    term = (u / flux['v_max'] - 1.0)
    return torch.where(
        u >= flux['v_max'],
        torch.zeros_like(u),
        torch.where(
            u <= -flux['v_max'],
            -flux['k_max'] * u,
            flux['v_max'] * flux['k_max'] * (term * term) / 4.0
        )
    )

def greenshield_Rp(u, flux):
    return torch.where(
        u >= flux['v_max'], 
        torch.tensor(0.0, device=u.device), 
        torch.where(
            u <= -flux['v_max'], 
            -flux['k_max'], 
            (flux['k_max'] / 2.0) * (u / flux['v_max'] - 1.0)
        )
    )

def greenberg_q(k, flux):
    k = torch.clamp(k, 0.0, flux['k_max'])
    return - flux['c0'] * k * torch.log(k + 1e-7) # k/kmax ?

def greenberg_qp(k, flux):
    return - flux['c0'] * (1.0 + torch.log(k + 1e-7))

def greenberg_R(u, flux):
    x = torch.exp(-u / flux['c0'] - 1.)
    return torch.where(
        flux['k_max'] < x,
        - flux['k_max'] * (flux["c0"] * np.log(flux['k_max']) + u),
        flux['c0'] * x
    )

def greenberg_Rp(u, flux):
    x = torch.exp(-u / flux['c0'] - 1.)
    return torch.where(
        flux['k_max'] < x,
        - flux['k_max'],
        - x)


# Define the breakpoints and corresponding slopes and intercepts
breakpoints = torch.tensor([
    0, 0.0069697, 0.0139394, 0.0209091, 0.0278788, 0.0348485, 0.0418182, 
    0.0487879, 0.0557576, 0.0627273, 0.06969697, 0.07666667, 0.08363636, 
    0.09060606, 0.09757576, 0.10454545, 0.11151515, 0.11848485, 0.12545455, 
    0.13242424, 0.13939394, 0.14636364, 0.15333333, 0.16030303, 0.16727273, 
    0.17424242, 0.18121212, 0.18818182, 0.19515152, 0.20212121, 0.20909091, 
    0.21606061, 0.2230303, 0.23, 0.2369697, 0.24393939, 0.25090909, 
    0.25787879, 0.26484848, 0.27181818, 0.27878788, 0.28575758, 0.29272727, 
    0.29969697, 0.30666667, 0.31363636, 0.32060606, 0.32757576, 0.33454545, 
    0.34151515, 0.34848485, 0.35545455, 0.36242424, 0.36939394, 0.37636364, 
    0.38333333, 0.39030303, 0.39727273, 0.40424242, 0.41121212, 0.41818182, 
    0.42515152, 0.43212121, 0.43909091, 0.44606061, 0.4530303, 0.46, 
    0.4669697, 0.47393939, 0.48090909, 0.48787879, 0.49484848, 0.50181818, 
    0.50878788, 0.51575758, 0.52272727, 0.52969697, 0.53666667, 0.54363636, 
    0.55060606, 0.55757576, 0.56454545, 0.57151515, 0.57848485, 0.58545455, 
    0.59242424, 0.59939394, 0.60636364, 0.61333333, 0.62030303, 0.62727273, 
    0.63424242, 0.64121212, 0.64818182, 0.65515152, 0.66212121, 0.66909091, 
    0.67606061, 0.6830303, 0.69
])

slopes = torch.tensor([
    -0.986435, -0.959645, -0.934962, -0.910621, -0.888048, -0.865815, -0.844977,
    -0.824497, -0.805114, -0.786117, -0.767972, -0.750250, -0.733174, -0.716559,
    -0.700417, -0.684775, -0.669455, -0.654676, -0.640087, -0.626078, -0.612144,
    -0.598826, -0.585482, -0.572792, -0.559982, -0.547861, -0.535538, -0.523917,
    -0.512081, -0.500883, -0.489524, -0.478702, -0.467783, -0.457307, -0.446795,
    -0.436641, -0.426507, -0.416652, -0.406871, -0.397295, -0.387844, -0.378527,
    -0.369385, -0.360313, -0.351460, -0.342617, -0.334037, -0.325410, -0.317087,
    -0.308663, -0.300583, -0.292351, -0.284500, -0.276450, -0.268801, -0.260954,
    -0.253473, -0.245836, -0.238507, -0.231068, -0.223883, -0.216634, -0.209586,
    -0.202518, -0.195601, -0.188705, -0.181914, -0.175182, -0.168511, -0.161936,
    -0.155380, -0.148956, -0.142509, -0.136231, -0.129888, -0.123749, -0.117508,
    -0.111502, -0.105357, -0.099480, -0.093428, -0.087661, -0.081726, -0.076046,
    -0.070233, -0.064633, -0.058937, -0.053415, -0.047831, -0.042384, -0.036908,
    -0.031533, -0.026162, -0.020857, -0.015588, -0.010350, -0.005179, -0.000684,
    0.000000
])

intercepts = torch.tensor([
    0.250001, 0.249814, 0.249470, 0.248961, 0.248332, 0.247557, 0.246686,
    0.245687, 0.244606, 0.243414, 0.242150, 0.240791, 0.239363, 0.237857,
    0.236282, 0.234647, 0.232939, 0.231187, 0.229357, 0.227502, 0.225560,
    0.223611, 0.221564, 0.219530, 0.217387, 0.215275, 0.213042, 0.210855,
    0.208546, 0.206282, 0.203907, 0.201569, 0.199134, 0.196724, 0.194233,
    0.191756, 0.189214, 0.186672, 0.184082, 0.181479, 0.178844, 0.176182,
    0.173505, 0.170786, 0.168072, 0.165298, 0.162547, 0.159721, 0.156937,
    0.154060, 0.151244, 0.148318, 0.145473, 0.142499, 0.139620, 0.136612,
    0.133693, 0.130658, 0.127696, 0.124637, 0.121632, 0.118550, 0.115505,
    0.112401, 0.109316, 0.106192, 0.103068, 0.099924, 0.096762, 0.093601,
    0.090402, 0.087223, 0.083988, 0.080794, 0.077522, 0.074313, 0.071007,
    0.067784, 0.064444, 0.061207, 0.057833, 0.054577, 0.051186, 0.047900,
    0.044496, 0.041179, 0.037764, 0.034416, 0.030991, 0.027612, 0.024178,
    0.020769, 0.017325, 0.013886, 0.010434, 0.006966, 0.003506, 0.000467, 0.000000
])

positive_breakpoints_p = torch.tensor([
    0.0069696969696969695, 0.013939393939393939, 0.02090909090909091,
    0.027878787878787878, 0.03484848484848485, 0.04181818181818182,
    0.04878787878787878, 0.055757575757575756, 0.06272727272727273,
    0.0696969696969697, 0.07666666666666666, 0.08363636363636363,
    0.0906060606060606, 0.09757575757575757, 0.10454545454545454,
    0.11151515151515151, 0.11848484848484848, 0.12545454545454546,
    0.13242424242424242, 0.1393939393939394, 0.14636363636363636,
    0.15333333333333332, 0.1603030303030303, 0.16727272727272727,
    0.17424242424242423, 0.1812121212121212, 0.18818181818181817,
    0.19515151515151513, 0.20212121212121212, 0.20909090909090908,
    0.21606060606060606, 0.22303030303030302, 0.23, 0.23696969696969697,
    0.24393939393939393, 0.2509090909090909, 0.2578787878787879,
    0.26484848484848483, 0.2718181818181818, 0.2787878787878788,
    0.28575757575757577, 0.2927272727272727, 0.2996969696969697,
    0.30666666666666664, 0.3136363636363636, 0.3206060606060606,
    0.3275757575757576, 0.33454545454545453, 0.3415151515151515,
    0.34848484848484845, 0.35545454545454547, 0.3624242424242424,
    0.3693939393939394, 0.37636363636363634, 0.3833333333333333,
    0.39030303030303026, 0.3972727272727273, 0.40424242424242424,
    0.4112121212121212, 0.41818181818181815, 0.4251515151515151,
    0.43212121212121213, 0.4390909090909091, 0.44606060606060605,
    0.453030303030303, 0.45999999999999996, 0.466969696969697,
    0.47393939393939394, 0.4809090909090909, 0.48787878787878786,
    0.4948484848484848, 0.5018181818181818, 0.5087878787878788,
    0.5157575757575757, 0.5227272727272727, 0.5296969696969697,
    0.5366666666666666, 0.5436363636363636, 0.5506060606060605,
    0.5575757575757576, 0.5645454545454546, 0.5715151515151515,
    0.5784848484848485, 0.5854545454545454, 0.5924242424242424,
    0.5993939393939394, 0.6063636363636363, 0.6133333333333333,
    0.6203030303030302, 0.6272727272727272, 0.6342424242424243,
    0.6412121212121212, 0.6481818181818182, 0.6551515151515152,
    0.6621212121212121, 0.6690909090909091, 0.676060606060606,
    0.683030303030303, 0.69
])

slopes_p = torch.tensor([
    3.609774, 3.684193, 3.527419, 3.356835, 3.214944, 3.087408, 2.957996,
    2.858621, 2.759245, 2.659871, 2.571903, 2.496725, 2.422573, 2.337011,
    2.284161, 2.216381, 2.164065, 2.110097, 2.048156, 2.003279, 1.952035,
    1.912639, 1.875742, 1.818880, 1.790716, 1.748425, 1.726946, 1.684121,
    1.648828, 1.620665, 1.584211, 1.556895, 1.543080, 1.498650, 1.484301,
    1.458052, 1.429891, 1.413089, 1.384391, 1.370042, 1.337958, 1.334435,
    1.302887, 1.284482, 1.267816, 1.248875, 1.234526, 1.220176, 1.204582,
    1.176417, 1.177664, 1.148966, 1.134615, 1.127532, 1.113360, 1.099009,
    1.085554, 1.077754, 1.061976, 1.045004, 1.040185, 1.020356, 1.008505,
    1.013449, 0.984752, 0.980876, 0.974277, 0.952713, 0.953247, 0.945042,
    0.934796, 0.918178, 0.913539, 0.899190, 0.899189, 0.884841, 0.884840,
    0.869826, 0.863585, 0.849365, 0.849105, 0.849235, 0.828383, 0.827039,
    0.814568, 0.813629, 0.799279, 0.799280, 0.794382, 0.778023, 0.782387,
    0.773660, 0.763674, 0.763674, 0.749325, 0.749325, 0.734976, 0.374744,
    0.000000
])

intercepts_p = torch.tensor([
    -0.998000, -0.998518, -0.996333, -0.992766, -0.988811, -0.984366,
    -0.978954, -0.974106, -0.968565, -0.962332, -0.956201, -0.950437,
    -0.944235, -0.936483, -0.931326, -0.924240, -0.918406, -0.912011,
    -0.904241, -0.898298, -0.891155, -0.885388, -0.879731, -0.870616,
    -0.865905, -0.858536, -0.854644, -0.846585, -0.839697, -0.834005,
    -0.826383, -0.820481, -0.817400, -0.807181, -0.803780, -0.797377,
    -0.790311, -0.785979, -0.778378, -0.774478, -0.765533, -0.764526,
    -0.755291, -0.749775, -0.744664, -0.738724, -0.734124, -0.729423,
    -0.724206, -0.714587, -0.715022, -0.704821, -0.699620, -0.697003,
    -0.691669, -0.686168, -0.680917, -0.677818, -0.671440, -0.664461,
    -0.662446, -0.654015, -0.648894, -0.651065, -0.638264, -0.636508,
    -0.633473, -0.623403, -0.623656, -0.619711, -0.614712, -0.606488,
    -0.604160, -0.596860, -0.596859, -0.589359, -0.589359, -0.581301,
    -0.577908, -0.570078, -0.569934, -0.570007, -0.558090, -0.557312,
    -0.550011, -0.549455, -0.540854, -0.540854, -0.537850, -0.527703,
    -0.530440, -0.524905, -0.518502, -0.518502, -0.509101, -0.509101,
    -0.499500, -0.255962, 0.000000
])

def underwood_q(k, flux):
    return 0.25 * np.exp(1) * k * torch.exp(-k)

def underwood_qp(k, flux):
    return 0.25 * np.exp(1) * (1 - k) * torch.exp(-k)

def underwood_R(u, flux):
    result = torch.empty_like(u)
    mask_neg = u <= 0
    mask_mid = (u > 0) & (u < 0.69 - 1e-7)
    mask_high = u >= 0.69 -1e-7

    result[mask_neg] = - u[mask_neg] + 0.25
    result[mask_high] = 0.0

    idx = torch.searchsorted(breakpoints, u[mask_mid], right=True) - 1
    result[mask_mid] = slopes[idx] * u[mask_mid] + intercepts[idx]
    return result

def underwood_Rp(u, flux):
    result = torch.empty_like(u)
    mask_neg = u <= 0
    mask_mid = (u > 0) & (u < 0.69 - 1e-7)
    mask_high = u >= 0.69 - 1e-7

    result[mask_neg] = -1.0
    result[mask_high] = 0.0
    if mask_mid.any():
        idx = torch.searchsorted(positive_breakpoints_p, u[mask_mid], right=True)
        result[mask_mid] = slopes_p[idx] * u[mask_mid] + intercepts_p[idx]
    return result

def trapezoidal_q(k, flux):
    return torch.where(
        k < flux['kcrit0'],
        flux['v_max'] * k,
        torch.where(
            k < flux['kcrit1'],
            flux['q_max'],
            flux['w'] * (k - flux['k_max']),
        )
    )

def trapezoidal_qp(k, flux):
    return torch.where(
        k < flux['kcrit0'],
        flux['v_max'],
        torch.where(
            k < flux['kcrit1'],
            torch.zeros_like(k),
            flux['w'],
        )
    )

def trapezoidal_R(u, flux):
    R1 = torch.where(
        u < flux['v_max'],
        (flux['v_max'] - u) * flux['kcrit0'],
        torch.zeros_like(u)
    )
    R2 = torch.where(
        u < 0,
        flux['q_max'] - u * flux['kcrit1'],
        flux['q_max'] - u * flux['kcrit0']
    )
    R3 = torch.where(
        u < flux['w'],
        - u * flux['k_max'],
        (flux['w'] - u) * flux['kcrit1'] - flux['k_max'] * flux['w']
    )
    return torch.maximum(torch.maximum(R1, R2), R3)

def trapezoidal_Rp(u, flux):
    R1 = torch.where(
        u < flux['v_max'],
        (flux['v_max'] - u) * flux['kcrit0'],
        torch.zeros_like(u)
    )
    R2 = torch.where(
        u < 0,
        flux['q_max'] - u * flux['kcrit1'],
        flux['q_max'] - u * flux['kcrit0']
    )
    R3 = torch.where(
        u < flux['w'],
        - u * flux['k_max'],
        (flux['w'] - u) * flux['kcrit1'] - flux['k_max'] * flux['w']
    )
    R1p = torch.where(
        u < flux['v_max'],
        -flux['kcrit0'],
        torch.zeros_like(u)
    )
    R2p = torch.where(
        u < 0,
        -flux['kcrit1'],
        -flux['kcrit0']
    )
    R3p = torch.where(
        u < flux['w'],
        -flux['k_max'],
        -flux['kcrit1']
    )

    return torch.where(
        (R1 >= R2) & (R1 >= R3),
        R1p,
        torch.where(
            R2 >= R3,
            R2p,
            R3p
        )
    )

def triangular_q(k, flux):
    return torch.where(
        k < flux['kcrit'],
        flux['v_max'] * k,
        flux['w'] * (k - flux['kmax'])
    )

def triangular_qp(k, flux):
    return torch.where(
        k < flux['kcrit'],
        flux['v_max'],
        flux['w']
    )

def triangular_R(u, flux):
    return torch.where(
        u >= flux['v_max'],
        0,
        torch.where(
            u <= flux['w'],
            -flux['kmax'] * u,
            flux['kcrit'] * (flux['v_max'] - u)
        )
    )

def triangular_Rp(u, flux):
    return torch.where(
        u >= flux['v_max'],
        0,
        torch.where(
            u <= flux['w'],
            -flux['kmax'],
            -flux['kcrit']
        )
    )

def trapezoidal_q1(k, flux):
    return torch.where(
        k < flux['kcrit0'],
        flux['v_max'] * k,
        torch.where(
            k < flux['kcrit1'],
            (flux['q_max1'] - flux['q_max2']) / (flux['kcrit0'] - flux['kcrit1']) * (k - flux['kcrit0']) + flux['q_max1'],
            flux['w'] * (k - flux['k_max']),
        )
    )


def trapezoidal_qp1(k, flux):
    return torch.where(
        k < flux['kcrit0'],
        flux['v_max'],
        torch.where(
            k < flux['kcrit1'],
            (flux['q_max1'] - flux['q_max2']) / (flux['kcrit0'] - flux['kcrit1']),
            flux['w'],
        )
    )


def trapezoidal_R1(u, flux):
    alpha = (flux['q_max1'] - flux['q_max2']) / (flux['kcrit0'] - flux['kcrit1'])
    return torch.where(
        u >= flux['v_max'],
        0,
        torch.where(
            u >= alpha,
            (flux['v_max'] - u) * flux['kcrit0'],
            torch.where(
                u <= flux['w'],
                -flux['kmax'] * u,
                alpha * (flux['kcrit1'] - flux['kcrit0']) + flux['q_max1'] - u * flux['kcrit1']
            )
        )
    )

def trapezoidal_Rp1(u, flux):
    alpha = (flux['q_max1'] - flux['q_max2']) / (flux['kcrit0'] - flux['kcrit1'])
    return torch.where(
        u >= flux['v_max'],
        0,
        torch.where(
            u >= alpha,
            - flux['kcrit0'],
            torch.where(
                u <= flux['w'],
                -flux['kmax'],
                - flux['kcrit1']
            )
        )
    )

trapezoidal1 = {
    'v_max': 1.,
    'w': -3/2.,
    'kmax': 1.,
    'k_max': 1.,
    'q_max1': .2,
    'q_max2': .3,
    'kcrit0': 0.2,
    'kcrit1': 0.8,
    'k_crit': .8,
    'q': trapezoidal_q1,
    'qp': trapezoidal_qp1,
    'R': trapezoidal_R1,
    'Rp': trapezoidal_Rp1
}

greenshield = {
    "v_max": 1,
    "w": -1,
    "q": greenshield_q,
    "qp": greenshield_qp,
    "R": greenshield_R,
    "Rp": greenshield_Rp,
    "k_max": 4.
}

greenshield1 = {
    "v_max": 1,
    "w": -1,
    "q": greenshield_q,
    "qp": greenshield_qp,
    "R": greenshield_R,
    "Rp": greenshield_Rp,
    "k_max": 1.,
    'kcrit': .5,
}

greenberg = {
    "v_max": - (np.log(0.0 + 1e-7) + 1) * 2.0, # 2.0 for c0
    "w": - (np.log(1.0 + 1e-7) + 1) * 2.0, # 2.0 for c0, 1.0 for kmax
    "q": greenberg_q,
    "qp": greenberg_qp,
    "R": greenberg_R,
    "Rp": greenberg_Rp,
    "k_max": 1.,
    "c0": 2.0,
    "kcrit": np.exp(-1)
}

underwood = {
    "v_max": 0.25 * np.exp(1) * (1),
    "w": 0,
    "q": underwood_q,
    "qp": underwood_qp,
    "R": underwood_R,
    "Rp": underwood_Rp,
    "k_max": 1.,
    "kcrit": 1.,
}

trapezoidal = {
    'v_max': 1.,
    'w': -1.,
    'k_max': 1.,
    'q_max': .2,
    'kcrit0': 0.2,
    'kcrit1': 0.8,
    'k_crit': .5,
    'q': trapezoidal_q,
    'qp': trapezoidal_qp,
    'R': trapezoidal_R,
    'Rp': trapezoidal_Rp
}

triangular = {
    'v_max': 1.,
    'w': -1.,
    'kmax': 1.,
    'kcrit': 0.5,
    'q': triangular_q,
    'qp': triangular_qp,
    'R': triangular_R,
    'Rp': triangular_Rp
}

skewed = {
    'v_max': 2.,
    'w': -1.,
    'kmax': 1.,
    'kcrit': 1/3,
    'q': triangular_q,
    'qp': triangular_qp,
    'R': triangular_R,
    'Rp': triangular_Rp
}

@force_float64
def Lax_Hopf_solver_Greenshield(ic_ks, dx =.1, dt=.1, Nx=100, Nt=140, flow=greenshield, device='cpu'):

    # Define initial conditions
    batch_size = ic_ks.shape[0]
    xmax = dx * Nx
    tmax = dt * Nt


    ic_xs = torch.linspace(0, xmax, ic_ks.shape[1]+1, device=device, dtype=torch.float64).unsqueeze(0).repeat(batch_size, 1)
    # ic_xs += torch.empty(batch_size, ic_xs.shape[1], device=device).uniform_(-(xmax/ic_ks.shape[1])/3, (xmax/ic_ks.shape[1])/3)
    # consider it as a problem on R
    ic_xs[:, 0] = -1e9
    ic_xs[:, -1] = 1e9

    ic_ks = ic_ks.to(device)

    # precompute bi's
    bi = []
    for i in range(ic_xs.shape[1] - 1):
        b = ic_ks[:, i] * ic_xs[:, i]
        for l in range(i):
            b -= (ic_xs[:, l + 1] - ic_xs[:, l]) * ic_ks[:, l]
        bi.append(b)
    bi = torch.stack(bi).to(device).T

    t = torch.arange(1e-9, tmax, dt, device=device, dtype=torch.float64).view(1, -1, 1, 1)
    x = torch.arange(0, xmax, dx, device=device, dtype=torch.float64).view(1, 1, -1, 1)

    xi = ic_xs.view(batch_size, 1, 1, -1)
    
    conditionJ = (xi > x - flow['v_max'] * t)
    conditionU = (xi < x - flow['w'] * t)

    Jl = torch.where(
        conditionJ.any(dim=-1), 
        torch.clamp(
            torch.argmax(conditionJ.float(), dim=-1) - 1, 
            min=0
        ), 
        len(xi)-2)*torch.ones_like(conditionJ.any(dim=-1)
    )
    
    Ju = torch.where(
        conditionU.any(dim=-1), 
        torch.clamp(
            xi.shape[-1] - torch.argmax(conditionU.int().flip(-1), dim=-1) - 1, 
            max=xi.shape[-1] - 2
        ), 
        torch.zeros_like(conditionU.any(dim=-1))
    )
    del conditionJ, conditionU

    xi = xi.expand(-1, t.shape[1], x.shape[2], -1)
    ki = ic_ks.view(batch_size, 1, 1, -1).expand(-1, t.shape[1], x.shape[2], -1)
    bi = bi.view(batch_size, 1, 1, -1).expand(-1, t.shape[1], x.shape[2], -1)

    max_range_len = (Ju - Jl + 1).max().item()
    i_range = Ju.unsqueeze(-1).expand(batch_size, t.shape[1], x.shape[2], max_range_len).clone()
    range_tensor = torch.arange(max_range_len, device=device).view(1, 1, 1, -1)
    i_range = torch.minimum(Jl.unsqueeze(-1) + range_tensor, Ju.unsqueeze(-1))
    del range_tensor, Ju, Jl

    xi_range_p1 = xi.gather(3, i_range + 1)
    xi_range = xi.gather(3, i_range)
    ki_range = ki.gather(3, i_range)
    bi_range = bi.gather(3, i_range)
    M_values = _Mc0(t, x, xi_range, xi_range_p1, ki_range, bi_range, flow)
    i_store = i_range.gather(3, torch.argmin(M_values, dim=-1, keepdim=True))
    del xi_range_p1, xi_range, ki_range, bi_range, M_values, i_range

    xip1 = xi.gather(3, i_store+1)
    xi = xi.gather(3, i_store)
    ki = ki.gather(3, i_store)
    del i_store
    rho = _rho_c0(t.squeeze(-1), x.squeeze(-1), xi.squeeze(-1), xip1.squeeze(-1), ki.squeeze(-1), flow=flow)
    return rho.cpu()

def _Mc0(t, x, xi, xip1, ki, bi, flow):
    c1 = xi + t * flow['w']
    c2 = xi + t * flow['qp'](ki, flow)
    c3 = xip1 + t * flow['qp'](ki, flow)
    # c4 = xip1 + t * flow['v_max']

    M = torch.where(
        (x >= c1) & (x < c2),
        t * flow['R']((x - xi) / t, flow) - ki * xi + bi,
        torch.where(
            (x >= c2) & (x < c3),
            t * flow['q'](ki, flux=flow) - ki * x + bi,
            t * flow['R']((x - xip1) / t, flow) - ki * xip1 + bi
            # torch.where(
            #     (x >= c3) & (x <= c4), 
            #     t * flow['R']((x - xip1) / t, flow) - ki * xip1 + bi, 
            #     torch.full_like(M_condition_1, float('inf')))
        )
    )

    return M

def _rho_c0(t, x, xi, xip1, ki, flow):
    t_flow_qp_ki = t * flow['qp'](ki, flux=flow)
    c1 = xi + t * flow['w']
    c2 = xi + t_flow_qp_ki
    c3 = xip1 + t_flow_qp_ki
    # c4 = xip1 + t * flow['v_max']

    return torch.where(
        (x >= c1) * (x < c2), 
        -flow['Rp']((x - xi)/t, flow),
        torch.where(
            (x >= c2) * (x < c3), 
            ki,
             -flow['Rp']((x - xip1)/t, flow)
            # torch.where(
            #     (x >= c3) * (x <= c4), 
            #     -flow['Rp']((x - xip1)/t, flow),
            #     torch.zeros_like(x)
            # )
        )
    )#.squeeze(-1)