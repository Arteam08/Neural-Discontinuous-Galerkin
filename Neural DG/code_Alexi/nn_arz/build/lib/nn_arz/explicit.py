import numpy as np
import matplotlib.pyplot as plt

import scipy

from tqdm import tqdm

def lambda_1_riemann(rho, v, a, b, gamma):
    return v - a * gamma * rho**gamma - b

def lambda_2_riemann(v):
    return v

# Rarefaction wave solver (1-rarefaction)
def rw1(x, t, rho_l, v_l, v_r, a, b, gamma):
    zheta = x / t
    if b == 0:
        # v = a**(-1/gamma) * (a*rho_l**gamma + v_l - v_r)**(1/gamma)
        v = (1 + 1/gamma)**(-1) * (v_l + a*rho_l**gamma + x/(gamma*t))
        rho = ((v - zheta) / (a * gamma))**(1/gamma)

        # Validity check
        validity_check = (v - v_l) - a * rho_l**gamma - b * np.log(rho_l) + a * rho**gamma + b * np.log(rho)
        if np.abs(validity_check) > 1e-10:
            print(f"Warning: Validity check failed with error {validity_check}")

        return rho, v

    c = (1 + 1/gamma)
    d = b / gamma
    f = b + zheta
    g = v_l + a*rho_l**gamma + b*np.log(rho_l) + (b + zheta)/gamma + b*np.log(a * gamma) / gamma
    
    # Solve for v using Lambert W function
    v = (d * scipy.special.lambertw(c * np.exp(g/d - c*f/d)/d) / c + f)
    
    # Solve for rho from v
    rho = ((v - b - zheta) / (a * gamma))**(1/gamma)
    
    # Validity check
    validity_check = (v - v_l) - a * rho_l**gamma - b * np.log(rho_l) + a * rho**gamma + b * np.log(rho)
    if np.abs(validity_check) > 1e-10:
        print(f"Warning: Validity check failed with error {validity_check}")

    return rho, v

def riemann_solution(rho_l, v_l, rho_r, v_r, a, b, gamma, nx, nt, x_min, x_max, t_max, verbose=True):
    # Compute the Riemann solution for the ARZ model
    # rho_l, v_l: left state
    # rho_r, v_r: right state
    # a, b, gamma: model parameters
    # Returns: rho, v

    # Initialize solution arrays
    rho = np.zeros((nx, nt))
    v = np.zeros((nx, nt))

    # Used to get x value from x index
    x_grid = np.linspace(x_min, x_max, nx)
    # Used to get t value from t index
    t_grid = np.linspace(0, t_max, nt)

    # Initial conditions
    rho[:, 0] = np.where(x_grid < 0, rho_l, rho_r)
    v[:, 0] = np.where(x_grid < 0, v_l, v_r)

    # Constants for rarefaction solution
    c = a * rho_l**gamma + b * np.log(rho_l) + v_l - v_r
    if b == 0:
        rho_star = a**(-1/gamma) * (a*rho_l**gamma + v_l - v_r)**(1/gamma)
    else:
        rho_star = gamma**(-1/gamma) * (b * scipy.special.lambertw(a * np.exp(c/b)**gamma * gamma/b)/a)**(1/gamma)

    # print(a*rho_star**gamma + b*np.log(rho_star) - (a*rho_l**gamma + b*np.log(rho_l) + v_l - v_r))
    v_star = v_r
    tho_2 = v_r

    ##### https://pubsonline.informs.org/doi/epdf/10.1287/trsc.1090.0283
    # v_max = max(v_l, v_r) # TODO not sure about that
    # rho_max = max(rho_l, rho_r) # TODO not sure about that
    # def p(rho):
    #     return rho**gamma
    
    # def Q_prime(rho):
    #     return (gamma + 1)*rho**gamma
    
    # with tqdm(range(1, nt), desc=f"t index 0/{nt}", disable=not(verbose)) as pbar:
    #     for t_index in pbar:
    #         t = t_grid[t_index]
    #         for x_index in range(nx):
    #             x = x_grid[x_index]

    #             rho_0 = Q_prime(v_r - v_l - p(rho_l))**(-1)
    #             q_0 = rho_0 * v_r
    #             q_l = rho_l * v_l

    #             zheta = x/t
    #             rho[x_index, t_index] = Q_prime(zheta - v_l - p(rho_l))**(-1)
    #             print(rho[x_index, t_index])
    #             v[x_index, t_index] = -p(rho[x_index, t_index]) + v_l + p(rho_l)

    #             # # Compute the Riemann solution at (x, t)
    #             # if v_r - v_l - p(rho_l) > v_max and v_l - gamma*rho_l**gamma <= 0: # Case 1.1 
    #             #     # TODO not checked
    #             #     print("Case 1.1")
    #             #     rho[x_index, t_index] = Q_prime(rho_l)**(-1) * (-v_l - p(rho_l))
    #             #     v[x_index, t_index] = -p(rho[x_index, t_index]) + v_l + p(rho_l)
    #             # elif v_r - v_l - p(rho_l) > v_max: # Case 1.2
    #             #     # TODO not checked
    #             #     print("Case 1.2")
    #             #     rho[x_index, t_index] = rho_l
    #             #     v[x_index, t_index] = v_l
    #             # elif 0 <= v_r - v_l - p(rho_l): # Case 2
    #             #     if v_r <= v_l: # Case 2.1
    #             #         if q_0 - q_l <= 0:
    #             #             print("Case 2.1.1")
    #             #             rho[x_index, t_index] = Q_prime(v_r - v_l - p(rho_l))**(-1)
    #             #             v[x_index, t_index] = v_r
    #             #         else:
    #             #             print("Case 2.1.2")
    #             #             rho[x_index, t_index] = rho_l
    #             #             v[x_index, t_index] = v_l
    #             #     else: # Case 2.2
    #             #         if v_l - gamma*rho_l**gamma >= 0: # Case 2.2.1
    #             #             print("Case 2.2.1")
    #             #             rho[x_index, t_index] = rho_l
    #             #             v[x_index, t_index] = v_l
    #             #         elif v_r - gamma*rho_0**gamma >= 0:
    #             #             print("Case 2.2.2")
    #             #             v[x_index, t_index] = v_r
    #             #             rho[x_index, t_index] = rho_0
    #             #         else:
    #             #             print("Case 2.2.3")
    #             #             rho[x_index, t_index] = Q_prime(-v_l - p(rho_l))**(-1)
    #             #             v[x_index, t_index] = -p(rho[x_index, t_index]) + v_l + p(rho_l)
    #             # else: # Case 3
    #             #     if v_r >= (rho_l/rho_max)*v_l:
    #             #         print("Case 3.1")
    #             #         v[x_index, t_index] = v_l
    #             #         rho[x_index, t_index] = rho_l
    #             #     else:
    #             #         print("Case 3.2")
    #             #         rho[x_index, t_index] = rho_max
    #             #         v[x_index, t_index] = v_r

                    
    #         pbar.set_description(f"t index {t_index + 1}/{nx}")

    # return rho, v

    #####
    print(lambda_1_riemann(rho_star, v_star, a, b, gamma))

    with tqdm(range(1, nt), desc=f"t index 0/{nt}", disable=not(verbose)) as pbar:
        for t_index in pbar:
            t = t_grid[t_index]
            for x_index in range(nx):
                x = x_grid[x_index]

                # Compute the Riemann solution at (x, t)
                if x < lambda_1_riemann(rho_l, v_l, a, b, gamma) * t:
                    rho[x_index, t_index] = rho_l
                    v[x_index, t_index] = v_l
                elif x <= lambda_1_riemann(rho_star, v_star, a, b, gamma) * t:
                    rho[x_index, t_index], v[x_index, t_index] = rw1(x, t, rho_l, v_l, v_r, a, b, gamma)
                elif x < tho_2 * t:
                    rho[x_index, t_index] = rho_star
                    v[x_index, t_index] = v_star
                else:
                    rho[x_index, t_index] = rho_r
                    v[x_index, t_index] = v_r

            pbar.set_description(f"t index {t_index + 1}/{nx}")

    return rho, v

