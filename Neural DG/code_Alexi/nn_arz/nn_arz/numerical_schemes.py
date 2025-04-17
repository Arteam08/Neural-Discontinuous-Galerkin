import numpy as np 

import torch

from tqdm import tqdm

LAX_FRIEDRICHS = 0
UPWIND = 1
MACCORMACK = 2
WENO = 3

def solve_old(rho, v, p, p_prime, nx, nt, delta_x, delta_t, scheme=0, epsilon=1e-10, verbose=True):
    # Creates temporary variables used for computation
    y = np.multiply(v + p(rho), rho)

    delta = delta_t/delta_x

    # TODO check CFL

    tol = 1e-6

    with tqdm(range(1, nt), desc=f"T", disable=not(verbose)) as pbar:
        for t in pbar:
            if scheme == LAX_FRIEDRICHS:
                def gudonov_flux(u_l, u_r, v): # https://www3.nd.edu/~yzhang10/WENO_ENO.pdf
                    fu_l = v[:-1]*u_l
                    fu_r = v[1:]*u_r 

                    mask_l = u_l > u_r

                    return np.where(mask_l, np.maximum(fu_l, fu_r), np.minimum(fu_r, fu_l))

                us_rho = weno(rho[:, t-1])
                us_y = weno(y[:, t-1])
                us_v = weno(v[:, t-1])

                # us_rho = rho[:, t-1]
                # us_y = y[:, t-1]
                # us_v = v[:, t-1]

                flux_rho = us_v * us_rho
                flux_y = us_v * us_y

                flux_rho = gudonov_flux(us_rho[:-1], us_rho[1:], us_v)
                flux_y = gudonov_flux(us_y[:-1], us_y[1:], us_v)

                rho[1:-1, t] = (rho[:-2, t-1] + rho[2:, t-1])/2 - delta * (np.roll(flux_rho, -1) - np.roll(flux_rho, 0))[1:]
                y[1:-1, t] = (y[:-2, t-1] + y[2:, t-1])/2 - delta * (np.roll(flux_y, -1) - np.roll(flux_y, 0))[1:]
                # rho[1:-1, t] = (rho[:-2, t-1] + rho[2:, t-1])/2 - delta * (us_v[2:]*us_rho[2:] - us_v[:-2]*us_rho[:-2])/2
                # y[1:-1, t] = (y[:-2, t-1] + y[2:, t-1])/2 - delta * (us_v[2:]*us_y[2:] - us_v[:-2]*us_y[:-2])/2


                rho[:, t] = np.where(rho[:, t] > tol, rho[:, t], 0)
                y[:, t] = np.where(rho[:, t] > tol, y[:, t], 0)


                # rho[1:-1, t] = (rho[:-2, t-1] + rho[2:, t-1])/2 - delta/2 * (rho[2:, t-1] * v[2:, t-1] - rho[:-2, t-1] * v[:-2, t-1])
                rho[:2, t], rho[-2:, t] = rho[2, t], rho[-3, t]

                # y[1:-1, t] = (y[:-2, t-1] + y[2:, t-1])/2 - delta/2 * (y[2:, t-1] * v[2:, t-1] - y[:-2, t-1] * v[:-2, t-1])
                y[:2, t], y[-2:, t] = y[2, t], y[-3, t]

                v[:, t] = np.where(rho[:, t] > tol, y[:, t] / rho[:, t] - p(rho[:, t]), 0)

                # rho[:, t] = np.where(v[:, t] < epsilon, 0, rho[:, t])
            elif scheme == WENO:
                def gudonov_flux(u_l, u_r, v): # https://www3.nd.edu/~yzhang10/WENO_ENO.pdf
                    fu_l = v[:-1]*u_l
                    fu_r = v[1:]*u_r 

                    mask_l = u_l > u_r

                    return np.where(mask_l, np.maximum(fu_l, fu_r), np.minimum(fu_r, fu_l))

                rho[:, t] = rho[:, t-1]
                y[:, t] = y[:, t-1]
                v[:, t] = v[:, t-1]
                for _ in range(2):
                    us_rho = weno(rho[:, t])
                    us_y = weno(y[:, t])
                    us_v = weno(v[:, t])

                    # us_rho = rho[:, t-1]
                    # us_y = y[:, t-1]
                    # us_v = v[:, t-1]

                    flux_rho = us_v * us_rho
                    flux_y = us_v * us_y

                    flux_rho = gudonov_flux(us_rho[:-1], us_rho[1:], us_v)
                    flux_y = gudonov_flux(us_y[:-1], us_y[1:], us_v)

                    rho[1:-1, t] = (rho[:-2, t] + 0*rho[1:-1, t] + rho[2:, t])/2 - delta * (np.roll(flux_rho, -1) - np.roll(flux_rho, 0))[1:]
                    y[1:-1, t] = (y[:-2, t] + 0*y[1:-1, t] + y[2:, t])/2 - delta * (np.roll(flux_y, -1) - np.roll(flux_y, 0))[1:]

                    rho[:, t] = np.where(rho[:, t] > tol, rho[:, t], 0)
                    y[:, t] = np.where(rho[:, t] > tol, y[:, t], 0)
                    
                    rho[:2, t], rho[-2:, t] = rho[2, t], rho[-3, t]
                    y[:2, t], y[-2:, t] = y[2, t], y[-3, t]

                    v[:, t] = np.where(rho[:, t] > tol, y[:, t] / rho[:, t] - p(rho[:, t]), 0)


                rho[:, t] = (rho[:, t] + rho[:, t-1])/2
                y[:, t] = (y[:, t] + y[:, t-1])/2   

                # rho[1:-1, t] = (rho[:-2, t-1] + rho[2:, t-1])/2 - delta * (us_v[2:]*us_rho[2:] - us_v[:-2]*us_rho[:-2])/2
                # y[1:-1, t] = (y[:-2, t-1] + y[2:, t-1])/2 - delta * (us_v[2:]*us_y[2:] - us_v[:-2]*us_y[:-2])/2


                rho[:, t] = np.where(rho[:, t] > tol, rho[:, t], 0)
                y[:, t] = np.where(rho[:, t] > tol, y[:, t], 0)

                rho[:2, t], rho[-2:, t] = rho[2, t], rho[-3, t]
                y[:2, t], y[-2:, t] = y[2, t], y[-3, t]

                v[:, t] = np.where(rho[:, t] > tol, y[:, t] / rho[:, t] - p(rho[:, t]), 0)

                # rho[:, t] = np.where(v[:, t] < epsilon, 0, rho[:, t])
            elif scheme == UPWIND: # TODO not verified
                rho[1:, t] = rho[1:, t-1] - delta * (rho[1:, t-1] * v[1:, t-1] - rho[:-1, t-1] * v[:-1, t-1])
                rho[0, t] = rho[1, t]

                y[1:, t] = y[1:, t-1] - delta * (y[1:, t-1] * v[1:, t-1] - y[:-1, t-1] * v[:-1, t-1])
                y[0, t] = y[1, t]

                v[:, t] = np.where(rho[:, t] > epsilon, y[:, t] / rho[:, t] - p(rho[:, t]), 0)
            elif scheme == MACCORMACK:
                rho_tilde = rho[1:, t-1] - delta * (rho[1:, t-1] * v[1:, t-1] - rho[:-1, t-1] * v[:-1, t-1])
                y_tilde = y[1:, t-1] - delta * (y[1:, t-1] * v[1:, t-1] - y[:-1, t-1] * v[:-1, t-1])
                v_tilde = np.where(rho_tilde > epsilon, y_tilde / rho_tilde - p(rho_tilde), 0)

                rho[1:-1, t] = (rho[1:-1, t-1] + rho_tilde[:-1])/2 - delta/2 * (rho_tilde[1:] * v_tilde[1:] - rho_tilde[:-1] * v_tilde[:-1])
                rho[0, t] = rho[1, t]
                rho[-1, t] = rho[-2, t]


                y[1:-1, t] = (y[1:-1, t-1] + y_tilde[:-1])/2 - delta/2 * (y_tilde[1:] * v_tilde[1:] - y_tilde[:-1] * v_tilde[:-1])
                y[0, t] = y[1, t]
                y[-1, t] = y[-2, t]

                v[:, t] = np.where(rho[:, t] > epsilon, y[:, t] / rho[:, t] - p(rho[:, t]), 0)
    return rho, v

# https://www3.nd.edu/~yzhang10/WENO_ENO.pdf

def weno(rho):
    # return rho
    n = len(rho)
    us = np.zeros(n)

    us[0] = 1 / 3 * rho[0] + 5 / 6 * rho[1] - 1 / 6 * rho[2]
    us[1] = -1 / 6 * rho[0] + 5 / 6 * rho[1] + 1 / 3 * rho[2]
    us[n-2] = -1 / 6 * rho[n-3] + 5 / 6 * rho[n-2] + 1 / 3 * rho[n-1]
    us[n-1] = 1 / 3 * rho[n-3] - 7 / 6 * rho[n-2] + 11 / 6 * rho[n-1]

    for i in range(2, n-2):
        # WENO reconstruction for three points left and right
        u1 = 1 / 3 * rho[i-2] - 7 / 6 * rho[i-1] + 11 / 6 * rho[i]
        u2 = -1 / 6 * rho[i-1] + 5 / 6 * rho[i] + 1 / 3 * rho[i+1]
        u3 = 1 / 3 * rho[i] + 5 / 6 * rho[i+1] - 1 / 6 * rho[i+2]

        # Weights calculation for WENO scheme
        gamma1, gamma2, gamma3 = 1 / 10, 3 / 5, 3 / 10
        eps = 1e-6

        beta1 = 13 / 12 * (rho[i-2] - 2 * rho[i-1] + rho[i]) ** 2 + 1 / 4 * (rho[i-2] - 4 * rho[i-1] + 3 * rho[i]) ** 2
        beta2 = 13 / 12 * (rho[i-1] - 2 * rho[i] + rho[i+1]) ** 2 + 1 / 4 * (rho[i-1] - rho[i+1]) ** 2
        beta3 = 13 / 12 * (rho[i] - 2 * rho[i+1] + rho[i+2]) ** 2 + 1 / 4 * (3 * rho[i] - 4 * rho[i+1] + rho[i+2]) ** 2

        alpha1 = gamma1 / (eps + beta1) ** 2
        alpha2 = gamma2 / (eps + beta2) ** 2
        alpha3 = gamma3 / (eps + beta3) ** 2

        w1 = alpha1 / (alpha1 + alpha2 + alpha3)
        w2 = alpha2 / (alpha1 + alpha2 + alpha3)
        w3 = alpha3 / (alpha1 + alpha2 + alpha3)

        # Final WENO flux reconstruction
        us[i] = w1 * u1 + w2 * u2 + w3 * u3
    return us
    # # Compute fluxes using the reconstructed values at interfaces
    # flows = np.zeros(n-1)
    # for i in range(n-1):
    #     flows[i] = godunov_flux(us[i], us[i+1])

    return us

def p(rho):
    return rho**2

def flux(u,v):
    # v = u[1] / u[0] - p(u[0])
    return v*u

def f(u, v):
    return v*u

def gudonov_flux(u_l, u_r, v): # TODO wrong ?
    # return 1/2 * ((flux(u_l,v) + flux(u_r,v)))
    # return flux(u_l)
    # if u_l >= u_r:
    #     return u_l*v
    # elif u_l <= 0 and u_r >= 0:
    #     return 0
    # else:
    #     return u_r*v
    if u_l <= u_r:
        return min(u_l*v, u_r*v)
    elif u_r < u_l:
        return u_l*v
    else:
        return u_r*v
    
def get_c1_c2(rho, y, v):
    # lbd1 = np.nanmax(abs(-3*rho**4 + 2*rho*y + np.sqrt(rho**5 * (rho**3 - 4*y)))/(2*rho**2))
    # lbd2 = np.nanmax(abs(-3*rho**4 + 2*rho*y - np.sqrt(rho**5 * (rho**3 - 4*y)))/(2*rho**2))
    lbd1 = np.max(abs(v))
    lbd2 = np.max(abs(v - 2*rho**2))

    c1 = lbd1 + 2e-3
    c2 = lbd2 + 1e-3
    return ((c1 if not np.isnan(c1) else 1e-3), (c2 if not np.isnan(c2) else 1e-3))



def solve_19(rho, v, p, p_prime, nx, nt, delta_x, delta_t, scheme=0, epsilon=1e-10, verbose=True):
    y = np.multiply(v + rho**2, rho)

    delta = delta_t/delta_x

    u = np.concatenate((rho[:,:,None], rho[:,:,None]*(v[:,:,None] + rho[:,:,None]**2)), axis=2)
    z = v[:,:,None] * u

    with tqdm(range(1, nt), desc=f"T", disable=not(verbose)) as pbar:
        for t in pbar:
            # us_g1 = np.concatenate((weno(g1[:, t-1, 0])[:,None], weno(g1[:, t-1, 1])[:,None]), axis=1)
            # us_g2 = np.concatenate((weno(g2[:, t-1, 0])[:,None], weno(g2[:, t-1, 1])[:,None]), axis=1)

            epsilon = 1e-8
            delta_t = 1e-5
            # C = np.array(get_c1_c2(rho[:, t-1], y[:, t-1], v=v[:, t-1]))

            # delta_t = delta_x / (max(max(C), 1)) #- 1e-5
            delta = delta_t/delta_x
            # print("C", C, "dt", delta_t)


            # Upwind
            # us_g1 = g1[1:, t-1]
            us_u = u[:, t-1] 
            # us_z = z[:, t-1]

            flow1 = v[:, t-1, None] * us_u
            # flow2 = us_u

            # s1 = -1/epsilon * (z[:, t-1] - f(u[:, t-1], v[:, [t-1]])) * 0
            # s2 = -1/epsilon * (z[:, t-1] - f(u[:, t-1], v[:, [t-1]]))

            # print(s2[1:-1])

            u[1:-1, t] = u[1:-1, t-1] - delta * (flow1[1:-1] - flow1[:-2]) #+ delta_x * delta_t * s1[1:-1] #* delta_x
            # z[1:-1, t] = z[1:-1, t-1] - delta * (flow2[2:] - flow2[1:-1]) + delta_x * delta_t * s2[1:-1] #* delta_x
            # Boundaries
            u[0, t] = u[1, t]
            u[-1, t] = u[-2, t]
            # z[0, t] = z[1, t]
            # z[-1, t] = z[-2, t]

            # u[:, t] = C**(-1)*(g1[:,t] - g2[:, t])/2
            # z[:, t] = (g1[:,t] + g2x[:,t]) / 2

            rho[:, t] = u[:, t, 0]
            y[:, t] = u[:, t, 1] 
            # print(abs((z[:, t, 0] / rho[:, t]) - z[:, t, 1] / y[:, t]).max())
            # exit()
            v[:, t] = np.where(rho[:, t], y[:, t] / rho[:, t] - rho[:, t]**2, 0) #* 0 + 1
            # y[:, t] = rho[:, t] * (v[:, t] + rho[:, t]**2)
            # v[:, t] = z[:, t, 1] / y[:, t] # np.where(rho[:, t] > 1e-12, u[:, t, 1] / rho[:, t] - rho[:, t]**2, 0)
            # v[:, t] = u[:, t, 1] / rho[:, t] - rho[:, t]**2
            # v[:, t] = np.where(rho[:, t] > epsilon, z[:, t, 0] / rho[:, t], 0)
            if t > 100:
                return rho[:,:t], v[:,:t]
            continue
            # us_rho = weno(rho[:, t-1])[:, np.newaxis]
            # us_v = weno(v[:, t-1])[:, np.newaxis]
            # # us_y = weno(y[:, t-1])[:, np.newaxis]

            us_u = np.concatenate((us_rho, us_v), axis=1)
            flows = np.zeros((nx-1, 2))
            # flowsl = np.zeros((nx-1, 2))
            # flowsr = np.zeros((nx-1, 2))
            for i in range(nx-1):
                flows[i] = gudonov_flux(us_u[i], us_u[i+1], v[i, t-1])
                # flowsl[i] = gudonov_flux(us_u[i], us_u[i])
                # flowsr[i] = gudonov_flux(us_u[i+1], us_u[i+1])

            rho[1:-1, t] = rho[1:-1, t-1] - delta * (flows[1:, 0] - flows[:-1, 0])
            y[1:-1, t] = y[1:-1, t-1] - delta * (flows[1:, 1] - flows[:-1, 1])
            # for i in range(1, nx-1):
            #     rho[i, t] = rho[i-1, t-1] - delta * (gudonov_flux(rho[i, t-1], rho[i+1, t-1], v[i,t-1]) - gudonov_flux(rho[i-1, t-1], rho[i, t-1], v[i,t-1]))
            #     y[i, t] = y[i-1, t-1] - delta * (gudonov_flux(y[i, t-1], y[i+1, t-1], v[i,t-1]) - gudonov_flux(y[i-1, t-1], y[i, t-1], v[i,t-1]))

            # v[:, t] = np.where(rho[:, t] > epsilon, y[:, t] / rho[:, t] - p(rho[:, t]), 0)

            # bounderies
            rho[0, t], rho[-1, t] = rho[1, t], rho[-2, t]
            # y[0, t], y[-1, t] = y[1, t], y[-2, t]
            v[0, t], v[-1, t] = v[1, t], v[-2, t]

            # if t > 1000:
            #     return rho[:,:t], v[:,:t]
    return rho, v

def solve_mine(rho, v, nx, nt, delta_x, delta_t, scheme=0, epsilon=1e-10, verbose=True):
    y = np.multiply(v + rho**2, rho)

    delta = delta_t/delta_x
    tt = 0
    u = np.concatenate((rho[:,:,None], rho[:,:,None]*(v[:,:,None] + rho[:,:,None]**2)), axis=2)
    z = v[:,:,None] * u

    time=[0]

    tol = 1e-6

    with tqdm(range(1, nt), desc=f"T", disable=not(verbose)) as pbar:
        for t in pbar:
            # us_g1 = np.concatenate((weno(g1[:, t-1, 0])[:,None], weno(g1[:, t-1, 1])[:,None]), axis=1)
            # us_g2 = np.concatenate((weno(g2[:, t-1, 0])[:,None], weno(g2[:, t-1, 1])[:,None]), axis=1)

            epsilon = 1e-3

            C = np.array(get_c1_c2(rho[:, t-1], y[:, t-1], v=v[:, t-1]))
            print("C", C)
            g1 = z + C * u
            g2 = z - C * u

            delta_t = .25 * delta_x / (max(max(C), 1))
            edt = delta_t/epsilon
            tt += delta_t
            print(tt)
            delta = delta_t/delta_x
            print("\t", delta, delta*max(C))
            # print("C", C, "dt", delta_t)


            # Upwind
            # us_g1 = np.roll(g1[:, t-1], 0, axis=0)
            # us_g2 = np.roll(g2[:, t-1], 0, axis=0)
            us_g1 = np.concatenate((weno(g1[:, t-1, 0])[:,None], weno(g1[:, t-1, 1])[:, None]), axis=1)#g1[:, t-1]
            us_g2 = np.concatenate((weno(g2[:, t-1, 0])[:,None], weno(g2[:, t-1, 1])[:, None]), axis=1)#g2[:, t-1]

            f1 = C * us_g1
            f2 = -C * us_g2

            g1[1:-1, t] = (g1[2:, t-1] + g1[:-2, t-1])/2 - delta * (f1[2:] - f1[:-2])/4
            g2[1:-1, t] = (g2[2:, t-1] + g2[:-2, t-1])/2 - delta * (f2[2:] - f2[:-2])/4

            g1[0, t], g1[-1, t] = g1[1, t], g1[-2, t]
            g2[0, t], g2[-1, t] = g2[1, t], g2[-2, t]

            z_star = (g1[:, t] + g2[:, t]) / 2
            u_star = C**(-1) * (g1[:, t] - g2[:, t]) / 2

            # v_star = np.where(u_star[:, 0] > tol, u_star[:, 1]/u_star[:, 0] - u_star[:, 0]**2, 0)
            v_star = np.where(u_star[:, 0] > tol, z_star[:, 0]/u_star[:, 0], 0)
            # v_star = np.where(u_star[:, 0] > tol, z_star[:, 1]/z_star[:, 0] - u_star[:, 0]**2, 0)

            s = -delta_t/epsilon * (z_star - u_star * v_star[:, None]) 
            print(u_star)
            if np.isnan(s).any():
                # print(s, u_star)
                exit()
            print(s.max())
            g1[:, t] += s#* delta_t
            g2[:, t] += s# * delta_t 

            g1[0, t], g1[-1, t] = g1[1, t], g1[-2, t]
            g2[0, t], g2[-1, t] = g2[1, t], g2[-2, t]

            u[:, t] = C**(-1)*(g1[:,t] - g2[:, t])/2
            z[:, t] = (g1[:,t] + g2[:,t]) / 2

            rho[:, t] = u[:, t, 0]
            rho = np.where(rho > tol, rho, 0)

            # v[:, t] = np.where(rho[:, t] > tol, u[:, t, 1]/rho[:, t] - rho[:, t]**2, 0)
            v[:, t] = np.where(rho[:, t] > tol, z[:, t, 0] / rho[:, t], 0)
            z[:, t, 0] = np.where(rho[:, t] > tol, z[:, t, 0], 0)
            z[:, t, 1] = np.where(rho[:, t] > tol, z[:, t, 1], 0)

            # print(rho[:, t].max(), rho[:, t].min())
            # print(abs((z[:, t, 0] / rho[:, t]) - z[:, t, 1] / y[:, t]).max())
            # # exit()
            # v[:, t] = z[:, t, 0] / rho[:, t] #* 0 + 1

            # y[:, t] = rho[:, t] * (v[:, t] + rho[:, t]**2)
            # v[:, t] = z[:, t, 1] / y[:, t] # np.where(rho[:, t] > 1e-12, u[:, t, 1] / rho[:, t] - rho[:, t]**2, 0)
            # v[:, t] = u[:, t, 1] / rho[:, t] - rho[:, t]**2
            # v[:, t] = np.where(rho[:, t] > epsilon, z[:, t, 0] / rho[:, t], 0)
            time.append(tt)
            if tt > 6. or t > 1000:
                print(tt)
                return rho[:,:t], v[:,:t], time[:-1]
            continue
            # us_rho = weno(rho[:, t-1])[:, np.newaxis]
            # us_v = weno(v[:, t-1])[:, np.newaxis]
            # # us_y = weno(y[:, t-1])[:, np.newaxis]

            us_u = np.concatenate((us_rho, us_v), axis=1)
            flows = np.zeros((nx-1, 2))
            # flowsl = np.zeros((nx-1, 2))
            # flowsr = np.zeros((nx-1, 2))
            for i in range(nx-1):
                flows[i] = gudonov_flux(us_u[i], us_u[i+1], v[i, t-1])
                # flowsl[i] = gudonov_flux(us_u[i], us_u[i])
                # flowsr[i] = gudonov_flux(us_u[i+1], us_u[i+1])

            rho[1:-1, t] = rho[1:-1, t-1] - delta * (flows[1:, 0] - flows[:-1, 0])
            y[1:-1, t] = y[1:-1, t-1] - delta * (flows[1:, 1] - flows[:-1, 1])
            # for i in range(1, nx-1):
            #     rho[i, t] = rho[i-1, t-1] - delta * (gudonov_flux(rho[i, t-1], rho[i+1, t-1], v[i,t-1]) - gudonov_flux(rho[i-1, t-1], rho[i, t-1], v[i,t-1]))
            #     y[i, t] = y[i-1, t-1] - delta * (gudonov_flux(y[i, t-1], y[i+1, t-1], v[i,t-1]) - gudonov_flux(y[i-1, t-1], y[i, t-1], v[i,t-1]))

            # v[:, t] = np.where(rho[:, t] > epsilon, y[:, t] / rho[:, t] - p(rho[:, t]), 0)

            # bounderies
            rho[0, t], rho[-1, t] = rho[1, t], rho[-2, t]
            # y[0, t], y[-1, t] = y[1, t], y[-2, t]
            v[0, t], v[-1, t] = v[1, t], v[-2, t]

            # if t > 1000:
            #     return rho[:,:t], v[:,:t]
    return rho, v





############
def timestep(v, u, dx, cfl=.25, crc1=1e-2, crc2=2e-2):
    c1 = np.abs(v - 2*u[:,0]**2)
    c2 = np.abs(v)

    c1max = np.max(c1)
    c2max = np.max(c2)

    cmax = np.max((c1max, c2max))

    c1 = cmax + crc1 # TODO really not C1/C2 ?
    c2 = cmax + crc2

    dt = (cfl*dx)/np.max((1, cmax+crc1))
    return dt, c1, c2

def flx(k, h1, h2, tol):
    h1 = np.where(h1 <= tol, 0, h1)
    # print("h1", h1.min())
    mask = h1 <= tol
    uu = np.zeros_like(h1)
    uu[~mask] = (h2[~mask] - h1[~mask]**3)/h1[~mask]  # y/rho - rho^2 = v
    # uu = np.where(uu <= tol, 0, uu) # y/rho - rho^2 = v
    # print("uu", uu.max(), uu.min())
    # h1 = np.where(h1 <= tol, 0, h1)
    # uu = np.where(h1 <= tol, 0, (h2 - h1*(h1**2))/h1) # y/rho - rho^2 = v

    if k == 0:
        flx = h1*uu
    else:
        flx = h1*uu*(uu + h1**2)

    return flx

def recon(qp, qm, i, k, c, epsilon, k_order=5):
    if k == 0 or k == 1:
        if k_order == 5: 
            return (0.5)*(poly(qp, i, 1, epsilon) - poly(qm, i+1, -1, epsilon))
        else:
            return (0.5/c)*(qp[i] - qm[i+1])
    else:
        if k_order == 5: 
            return (0.5)*(poly(qp, i, 1, epsilon) + poly(qm, i+1, -1, epsilon))
        else:
            return (0.5)*(qp[i] + qm[i+1])
        
def recon_vect(qp, qm, k, c, epsilon, k_order=5):
    if k == 0 or k == 1:
        if k_order == 5: 
            return (0.5/c)*(poly_vect(qp, 1, epsilon) - poly_vect(np.roll(qm, -1), -1, epsilon)) #TODO WARNING THIS WAS -1 
        else:
            return (0.5/c)*(qp - np.roll(qm, -1)) #TODO WARNING THIS WAS -1
    else:
        if k_order == 5: 
            return (0.5)*(poly_vect(qp, 1, epsilon) + poly_vect(np.roll(qm, -1), -1, epsilon)) #TODO WARNING THIS WAS -1
        else:
            return (0.5)*(qp + np.roll(qm, -1)) #TODO WARNING THIS WAS -1

def poly(q, i, lr, epsilon): 
    s0 = (13/12)*(q[i] - 2*q[i+1] + q[i+2])**2 + 0.25*(3.*q[i] - 4.*q[i+1] + q[i+2])**2
    s1 = (13/12)*(q[i-1] - 2*q[i] + q[i+1])**2 + 0.25*(q[i-1] - q[i+1])**2
    s2 = (13/12)*(q[i-2] - 2*q[i-1] + q[i])**2 + 0.25*(q[i-2] - 4*q[i-1] + 3*q[i])**2

    eta5 = np.abs(s0 - s2)
    d0 = 0.3
    d1 = 0.6
    d2 = 0.1

    # Banda and Seaid
    # a0 = d0*(1/(s0+epsilon)**2)
    # a1 = d1*(1/(s1+epsilon)**2)
    # a2 = d2*(1/(s2+epsilon)**2)

    # corrected Chen et al
    a0 = d0*(1 + eta5/(s0+epsilon)**2)
    a1 = d1*(1 + eta5/(s1+epsilon)**2)
    a2 = d2*(1 + eta5/(s2+epsilon)**2)

    suma = a0 + a1 + a2

    w0 = a0/suma
    w1 = a1/suma
    w2 = a2/suma

    if lr == 1:
        q0 = (1/6)*(-q[i+2] + 5*q[i+1] + 2*q[i])
        q1 = (1/6)*(-q[i-1] + 5*q[i] + 2*q[i+1])
        q2 = (1/6)*(2*q[i-2] - 7*q[i-1] + 11*q[i])
    else:
        q0 = (1/6)*(2*q[i+2] - 7*q[i+1] + 11*q[i])
        q1 = (1/6)*(-q[i+1] + 5*q[i] + 2*q[i-1])
        q2 = (1/6)*(-q[i-2] + 5*q[i-1] + 2*q[i])
 
    return w0*q0 + w1*q1 + w2*q2

def poly_vect(q, lr, epsilon): 
    # epsilon=1e-40
    q_m_1 = np.roll(q, 1)
    q_m_2 = np.roll(q, 2)
    q_p_1 = np.roll(q, -1)
    q_p_2 = np.roll(q, -2)
    s0 = (13/12)*(q - 2*q_p_1 + q_p_2)**2 + 0.25*(3.*q - 4.*q_p_1 + q_p_2)**2
    s1 = (13/12)*(q_m_1 - 2*q + q_p_1)**2 + 0.25*(q_m_1 - q_p_1)**2
    s2 = (13/12)*(q_m_2 - 2*q_m_1 + q)**2 + 0.25*(q_m_2 - 4*q_m_1 + 3*q)**2

    eta5 = np.abs(s0 - s2)
    d0 = 0.3
    d1 = 0.6
    d2 = 0.1

    # Banda and Seaid
    # a0 = d0*(1/(s0+epsilon)**2)
    # a1 = d1*(1/(s1+epsilon)**2)
    # a2 = d2*(1/(s2+epsilon)**2)

    # corrected Chen et al
    a0 = d0*(1 + eta5/(s0+epsilon)**1)
    a1 = d1*(1 + eta5/(s1+epsilon)**1)
    a2 = d2*(1 + eta5/(s2+epsilon)**1)

    suma = a0 + a1 + a2

    w0 = a0/suma
    w1 = a1/suma
    w2 = a2/suma
    # print(max(s0), max(s1), max(s2), max(eta5), max(a0), max(a1), max(a2), max(w0), max(w1), max(w2))

    if lr == 1:
        q0 = (1/6)*(-q_p_2 + 5*q_p_1 + 2*q)
        q1 = (1/6)*(-q_m_1 + 5*q + 2*q_p_1)
        q2 = (1/6)*(2*q_m_2 - 7*q_m_1 + 11*q)
    else:
        q0 = (1/6)*(2*q_p_2 - 7*q_p_1 + 11*q)
        q1 = (1/6)*(-q_p_1 + 5*q + 2*q_m_1)
        q2 = (1/6)*(-q_m_2 + 5*q_m_1 + 2*q)

    return w0*q0 + w1*q1 + w2*q2

def u_cal(rr, v, tol):
    mask = rr[:, 0] <= tol
    rr[mask, 0] = 0
    #rr[mask, 1] = 0
    v[mask] = 0
    v[~mask] = (rr[~mask, 1] - rr[~mask, 0]**3)/rr[~mask, 0]  # y/rho - rho^2 = v

    return rr, v

def solve(rho, v, nx, delta_x, tol, max_time, epsilon=1e-10, verbose=True):
    # Runge-Kutta parameters       
    gamma1 = (3. + np.sqrt(3.))/6.
    ab21 = gamma1
    ab31 = gamma1 - 1.
    ab32 = 2. - 2.*gamma1
    a22 = gamma1
    a32 = 1. - 2.*gamma1
    a33 = gamma1
    #####
    
    y = rho*(v + rho**2)
    u = np.concatenate((
        rho[:,:,None], 
        y[:,:,None],
        flx(0, rho, v, tol)[:,:,None],
        flx(1, rho, v, tol)[:,:,None]
        ), axis=2)
    u0 = u[:, 0]
    v_current = v[:, 0]

    delta_t, c1, c2 = timestep(v_current, u0, delta_x, cfl=.05, crc1=1e-2, crc2=1e-2)
    c = np.array([c1, c2])


    edt = delta_t / epsilon # TODO what is this ?
    time = 0
    t = 0
    tt = [t]
    # IMEX Runge-KUTTA (3rd-order)  
    while True:
        # CHARACTERISTIC VARIABLES
        qp = u0[:, 2:] + c*u0[:, :2]
        qm = u0[:, 2:] - c*u0[:, :2]
        if  qp.max() > 1e1:
            return rho[:, :t], v[:, :t], tt
        u2s = np.zeros((nx, 4))
        ff1 = np.zeros((nx, 4))

        ff1[:, 0] = (-(recon_vect(qp[:, 0], qm[:, 0], 2, c1, epsilon) - recon_vect(np.roll(qp[:,0], 1), np.roll(qm[:,0], 1), 2, c1, epsilon))/delta_x)
        ff1[:, 1] = (-(recon_vect(qp[:, 1], qm[:, 1], 3, c2, epsilon) - recon_vect(np.roll(qp[:,1], 1), np.roll(qm[:,1], 1), 3, c2, epsilon))/delta_x)
        ff1[:, 2] = -c1**2*((recon_vect(qp[:, 0], qm[:, 0], 0, c1, epsilon) - recon_vect(np.roll(qp[:,0], 1), np.roll(qm[:,0], 1), 0, c1, epsilon))/delta_x)
        ff1[:, 3] = -c2**2*((recon_vect(qp[:, 1], qm[:, 1], 1, c2, epsilon) - recon_vect(np.roll(qp[:,1], 1), np.roll(qm[:,1], 1), 1, c2, epsilon))/delta_x)

        u2s = u0 + delta_t * ab21 * ff1
        u2 = u2s
        u2[:, 2] = (u2s[:, 2] + edt*a22*flx(0, u2[:, 0], u2[:, 1], tol))/(1. + edt*a22)
        u2[:, 3] = (u2s[:, 3] + edt*a22*flx(1, u2[:, 0], u2[:, 1], tol))/(1. + edt*a22)

        u2[:, 0] = np.where(u2[:, 0] <= tol, 0, u2[:, 0])
        u2[:, 1] = np.where(u2[:, 0] <= tol, 0, u2[:, 1])

        u2[:2, :2] = u2[[2], :2]
        u2[-2:, :2] = u2[[-3], :2] 

        for k in [2, 3]:
            u2[:2, k] = flx(k-2, u2[:2, 0], u2[:2, 1], tol)
            u2[-2:, k] = flx(k-2, u2[-2:, 0], u2[-2:, 1], tol)

        u2, v_current = u_cal(u2, v_current, tol)

        qp2 = u2[:, 2:] + c*u2[:, :2]
        qm2 = u2[:, 2:] - c*u2[:, :2]


        ff1 = np.zeros((nx, 4))
        ff2 = np.zeros((nx, 4))

        ff1[:, 0] = (-(recon_vect(qp[:, 0], qm[:, 0], 2, c1, epsilon) - recon_vect(np.roll(qp[:,0], 1), np.roll(qm[:,0], 1), 2, c1, epsilon))/delta_x)
        ff1[:, 1] = (-(recon_vect(qp[:, 1], qm[:, 1], 3, c2, epsilon) - recon_vect(np.roll(qp[:,1], 1), np.roll(qm[:,1], 1), 3, c2, epsilon))/delta_x)
        ff1[:, 2] = -c1**2*((recon_vect(qp[:, 0], qm[:, 0], 0, c1, epsilon) - recon_vect(np.roll(qp[:,0], 1), np.roll(qm[:,0], 1), 0, c1, epsilon))/delta_x)
        ff1[:, 3] = -c2**2*((recon_vect(qp[:, 1], qm[:, 1], 1, c2, epsilon) - recon_vect(np.roll(qp[:,1], 1), np.roll(qm[:,1], 1), 1, c2, epsilon))/delta_x)

        ff2[:, 0] = (-(recon_vect(qp2[:, 0], qm2[:, 0], 2, c1, epsilon) - recon_vect(np.roll(qp2[:,0], 1), np.roll(qm2[:,0], 1), 2, c1, epsilon))/delta_x)
        ff2[:, 1] = (-(recon_vect(qp2[:, 1], qm2[:, 1], 3, c2, epsilon) - recon_vect(np.roll(qp2[:,1], 1), np.roll(qm2[:,1], 1), 3, c2, epsilon))/delta_x)
        ff2[:, 2] = -c1**2*((recon_vect(qp2[:, 0], qm2[:, 0], 0, c1, epsilon) - recon_vect(np.roll(qp2[:,0], 1), np.roll(qm2[:,0], 1), 0, c1, epsilon))/delta_x)
        ff2[:, 3] = -c2**2*((recon_vect(qp2[:, 1], qm2[:, 1], 1, c2, epsilon) - recon_vect(np.roll(qp2[:,1], 1), np.roll(qm2[:,1], 1), 1, c2, epsilon))/delta_x)

        u3s = np.zeros((nx, 4))
        u3s = u0 + delta_t*ab31*ff1 + delta_t*ab32*ff2

        u3 = u3s
        for k in [2, 3]:
            u3[:, k] = (u3s[:, k] - edt*a32*(u2[:, k] - flx(k-2, u2[:, 0], u2[:, 1], tol)) + edt*a33*flx(k-2, u3[:, 0], u3[:, 1], tol))/(1. + edt*a33)

        u3[:, 0] = np.where(u3[:, 0] <= tol, 0, u3[:, 0])
        u3[:, 1] = np.where(u3[:, 0] <= tol, 0, u3[:, 1])
            
        # Bountary Values for l=3 RK stage 

        # Neuman
        u3[:2, :2] = u3[[2], :2]
        u3[-2:, :2] = u3[[-2], :2] 

        for k in [2, 3]:
            u3[:2, k] = flx(k-2, u3[:2, 0], u3[:2, 1], tol)
            u3[-2:, k] = flx(k-2, u3[-2:, 0], u3[-2:, 1], tol)

        u3, v_current = u_cal(u3, v_current, tol)

        qp3 = u3[:, 2:] + c*u3[:, :2]
        qm3 = u3[:, 2:] - c*u3[:, :2]

        ff2 = np.zeros((nx, 4))
        ff3 = np.zeros((nx, 4))

        ff2[:, 0] = (-(recon_vect(qp2[:, 0], qm2[:, 0], 2, c1, epsilon) - recon_vect(np.roll(qp2[:,0], 1), np.roll(qm2[:,0], 1), 2, c1, epsilon))/delta_x)
        ff2[:, 1] = (-(recon_vect(qp2[:, 1], qm2[:, 1], 3, c2, epsilon) - recon_vect(np.roll(qp2[:,1], 1), np.roll(qm2[:,1], 1), 3, c2, epsilon))/delta_x)
        ff2[:, 2] = -c1**2*((recon_vect(qp2[:, 0], qm2[:, 0], 0, c1, epsilon) - recon_vect(np.roll(qp2[:,0], 1), np.roll(qm2[:,0], 1), 0, c1, epsilon))/delta_x)
        ff2[:, 3] = -c2**2*((recon_vect(qp2[:, 1], qm2[:, 1], 1, c2, epsilon) - recon_vect(np.roll(qp2[:,1], 1), np.roll(qm2[:,1], 1), 1, c2, epsilon))/delta_x)

        ff3[:, 0] = (-(recon_vect(qp3[:, 0], qm3[:, 0], 2, c1, epsilon) - recon_vect(np.roll(qp3[:,0], 1), np.roll(qm3[:,0], 1), 2, c1, epsilon))/delta_x)
        ff3[:, 1] = (-(recon_vect(qp3[:, 1], qm3[:, 1], 3, c2, epsilon) - recon_vect(np.roll(qp3[:,1], 1), np.roll(qm3[:,1], 1), 3, c2, epsilon))/delta_x)
        ff3[:, 2] = -c1**2*((recon_vect(qp3[:, 0], qm3[:, 0], 0, c1, epsilon) - recon_vect(np.roll(qp3[:,0], 1), np.roll(qm3[:,0], 1), 0, c1, epsilon))/delta_x)
        ff3[:, 3] = -c2**2*((recon_vect(qp3[:, 1], qm3[:, 1], 1, c2, epsilon) - recon_vect(np.roll(qp3[:,1], 1), np.roll(qm3[:,1], 1), 1, c2, epsilon))/delta_x)

        u0[:, :2] += delta_t*(ff2[:, :2] + ff3[:, :2])/2.
        for k in [2, 3]:
            u0[:, k] += 0.5*delta_t*(ff2[:, k] + ff3[:, k]) - (
                0.5*edt*(u2[:, k] - flx(k-2, u2[:, 0], u2[:, 1], tol)) 
                + 0.5*edt*(u3[:, k] - flx(k-2, u3[:, 0], u3[:, 1], tol)))

        u0[:, 0] = np.where(u0[:, 0] <= tol, 0, u0[:, 0])
        u0[:, 1] = np.where(u0[:, 0] <= tol, 0, u0[:, 1])


        # Bountary Values for time evolved at (n+1) solution values
        u0[:2, :2] = u0[[2], :2]
        u0[-2:, :2] = u0[[-3], :2] # 2.*u0[-3, :2] - u0[-3, :2] because -> U0(K,NPTS+1)= U0(K,NPTS) !2.d0*U0(K,NPTS)-U0(K,NPTS)
    
        for k in [2, 3]:
            u0[:2, k] = flx(k-2, u0[:2, 0], u0[:2, 1], tol)
            u0[-2:, k] = flx(k-2, u0[-2:, 0], u0[-2:, 1], tol)     

        u0, v_current = u_cal(u0, v_current, tol)
                # Bountary Values for time evolved at (n+1) solution values
        u0[:2, :2] = u0[[2], :2]
        u0[-2:, :2] = u0[[-3], :2] # 2.*u0[-3, :2] - u0[-3, :2] because -> U0(K,NPTS+1)= U0(K,NPTS) !2.d0*U0(K,NPTS)-U0(K,NPTS)
    
        for k in [2, 3]:
            u0[:2, k] = flx(k-2, u0[:2, 0], u0[:2, 1], tol)
            u0[-2:, k] = flx(k-2, u0[-2:, 0], u0[-2:, 1], tol)     


        t+=1
        # rho[:, t] = u0[:, 0]
        v[:, t] = v_current

        time += delta_t

        qp = u0[:, 2:] + c*u0[:, :2]
        qm = u0[:, 2:] - c*u0[:, :2]

        rrecp = np.zeros(nx)
        urecp = np.zeros(nx)
        for i in range(2, nx-3):
            rrecp[i] = recon(qp[:, 0], qm[:, 0], i, 0, c1, epsilon)
            urecp[i] = recon(qp[:, 1], qm[:, 1], i, 1, c2, epsilon)

        delta_t, c1, c2 = timestep(v_current, u0, delta_x, cfl=.2, crc1=5e-3, crc2=1.e-3)
        c = np.array([c1, c2])
        edt = delta_t / epsilon 
        tt.append(time)

        rho[:, t] = u0[:, 0]
        v[:, t] = v_current

        print(time, max_time)
        if time > max_time:
            # print(rho[:, t], v[:, t])
            print("HERE NOW")
            return rho[:, :t+1], v[:, :t+1], tt

def solve_____(rho, v, nx, delta_x, tol, max_time, epsilon=1e-10, verbose=True):
    y = np.multiply(v + rho**2, rho)
    nt = 10000
    delta_t = 1e-3
    print(delta_x)
    delta = delta_t/delta_x

    u = np.concatenate((rho[:,:,None], rho[:,:,None]*(v[:,:,None] + rho[:,:,None]**2)), axis=2)

    with tqdm(range(1, nt), desc=f"T", disable=not(verbose)) as pbar:
        for t in pbar:
            epsilon = 1e-8
            u[:, t] =  (np.roll(u[:,t-1], 1) + np.roll(u[:,t-1], -1))/2 + delta * (np.roll(v[:,[t-1]]*u[:, t-1], -1) - np.roll(v[:,[t-1]]*u[:, t-1], 1))/2
            #(np.roll(u[:,t-1], 1) + np.roll(u[:,t-1], -1))/2 \
            
            
            rho[:, t] = u[:, t, 0]
            v[:, t] = np.where(rho[:, t], u[:, t, 1]/rho[:, t] - rho[:, t]**2, 0)
    
            rho[:4, t] = rho[4, t]
            rho[-4:, t] = rho[-5, t]
            v[:4, t] = v[4, t]
            v[-4:, t] = v[-5, t] 
            # if t>50:
            #     print(rho[:, t])
            #     exit()
    v = np.where(v < 0, 0, v)
    v = np.where(v > 1, 1, v)
    return rho, v


def timestep_modified(v, u, dx, cfl=.25, crc1=1e-2, crc2=2e-2):
    c1 = torch.abs(v - 2*u[:, :, 0]**2)
    c2 = torch.abs(v)

    c1max = torch.max(c1, dim=-1).values
    c2max = torch.max(c2, dim=-1).values

    cmax = torch.maximum(c1max, c2max)

    c1 = cmax + crc1 # TODO really not C1/C2 ?
    c2 = cmax + crc2

    dt = (cfl*dx)/torch.maximum(torch.ones_like(cmax), cmax+crc1)
    return dt, c1, c2

def flx_modified(k, h1, h2, tol):
    h1 = torch.where(h1 <= tol, 0, h1)
    # print("h1", h1.min())
    mask = h1 <= tol
    uu = torch.zeros_like(h1)
    uu[~mask] = (h2[~mask] - h1[~mask]**3)/h1[~mask]  # y/rho - rho^2 = v
    # uu = np.where(uu <= tol, 0, uu) # y/rho - rho^2 = v
    # print("uu", uu.max(), uu.min())
    # h1 = np.where(h1 <= tol, 0, h1)
    # uu = np.where(h1 <= tol, 0, (h2 - h1*(h1**2))/h1) # y/rho - rho^2 = v

    if k == 0:
        flx = h1*uu
    else:
        flx = h1*uu*(uu + h1**2)

    return flx

def recon_vect_modified(qp, qm, k, c, epsilon, k_order=5):
    if k == 0 or k == 1:
        if k_order == 5: 
            return (0.5/c.unsqueeze(-1))*(poly_vect_modified(qp, 1, epsilon) - poly_vect_modified(torch.roll(qm, -1, -1), -1, epsilon)) #TODO WARNING THIS WAS -1 
        else:
            return (0.5/c.unsqueeze(-1))*(qp - torch.roll(qm, -1, -1)) #TODO WARNING THIS WAS -1
    else:
        if k_order == 5: 
            return (0.5)*(poly_vect_modified(qp, 1, epsilon) + poly_vect_modified(torch.roll(qm, -1, -1), -1, epsilon)) #TODO WARNING THIS WAS -1
        else:
            return (0.5)*(qp + torch.roll(qm, -1, -1)) #TODO WARNING THIS WAS -1

def poly_vect_modified(q, lr, epsilon): 
    # epsilon=1e-40
    q_m_1 = torch.roll(q, 1, -1)
    q_m_2 = torch.roll(q, 2, -1)
    q_p_1 = torch.roll(q, -1, -1)
    q_p_2 = torch.roll(q, -2, -1)
    s0 = (13/12)*(q - 2*q_p_1 + q_p_2)**2 + 0.25*(3.*q - 4.*q_p_1 + q_p_2)**2
    s1 = (13/12)*(q_m_1 - 2*q + q_p_1)**2 + 0.25*(q_m_1 - q_p_1)**2
    s2 = (13/12)*(q_m_2 - 2*q_m_1 + q)**2 + 0.25*(q_m_2 - 4*q_m_1 + 3*q)**2

    eta5 = torch.abs(s0 - s2)
    d0 = 0.3
    d1 = 0.6
    d2 = 0.1

    # Banda and Seaid
    # a0 = d0*(1/(s0+epsilon)**2)
    # a1 = d1*(1/(s1+epsilon)**2)
    # a2 = d2*(1/(s2+epsilon)**2)

    # corrected Chen et al
    a0 = d0*(1 + eta5/(s0+epsilon)**1)
    a1 = d1*(1 + eta5/(s1+epsilon)**1)
    a2 = d2*(1 + eta5/(s2+epsilon)**1)

    suma = a0 + a1 + a2

    w0 = a0/suma
    w1 = a1/suma
    w2 = a2/suma
    # print(max(s0), max(s1), max(s2), max(eta5), max(a0), max(a1), max(a2), max(w0), max(w1), max(w2))

    if lr == 1:
        q0 = (1/6)*(-q_p_2 + 5*q_p_1 + 2*q)
        q1 = (1/6)*(-q_m_1 + 5*q + 2*q_p_1)
        q2 = (1/6)*(2*q_m_2 - 7*q_m_1 + 11*q)
    else:
        q0 = (1/6)*(2*q_p_2 - 7*q_p_1 + 11*q)
        q1 = (1/6)*(-q_p_1 + 5*q + 2*q_m_1)
        q2 = (1/6)*(-q_m_2 + 5*q_m_1 + 2*q)

    return w0*q0 + w1*q1 + w2*q2

def u_cal_modified(rr, v, tol):
    mask = rr[:, :, 0] <= tol

    mask0 = mask.unsqueeze(-1).repeat(1, 1, 4)
    mask0[:, :, 1:] = False
    n_mask0 = (~mask).unsqueeze(-1).repeat(1, 1, 4)
    n_mask0[:, :, 1:] = False
    n_mask1 = (~mask).unsqueeze(-1).repeat(1, 1, 4)
    n_mask1[:, :, 0] = False
    n_mask1[:, :, 2:] = False
    rr[mask0] = 0
    #rr[mask, 1] = 0
    v[mask] = 0
    v[~mask] = (rr[n_mask1] - rr[n_mask0]**3)/rr[n_mask0]  # y/rho - rho^2 = v

    return rr, v

def solve_modified(rho, v, nx, delta_x, tol, max_time, epsilon=1e-10, dt=None, verbose=True):
    # Runge-Kutta parameters       
    gamma1 = (3. + np.sqrt(3.))/6.
    ab21 = gamma1
    ab31 = gamma1 - 1.
    ab32 = 2. - 2.*gamma1
    a22 = gamma1
    a32 = 1. - 2.*gamma1
    a33 = gamma1
    #####

    batch_size = rho.shape[0]
    
    y = rho*(v + rho**2)
    u = torch.stack((
        rho, 
        y,
        flx_modified(0, rho, v, tol),
        flx_modified(1, rho, v, tol)
        ), dim=-1)
    u0 = u[:, :, 0]
    v_current = v[:, :, 0]

    delta_t, c1, c2 = timestep_modified(v_current, u0, delta_x, cfl=.05, crc1=1e-2, crc2=1e-2)
    if dt is not None:
        delta_t = dt
    c = torch.stack([c1, c2]).to(rho.device).T.unsqueeze(1)


    edt = delta_t / epsilon # TODO what is this ?
    time = 0
    t = 0
    tt = [t]
    # IMEX Runge-KUTTA (3rd-order)  
    while True:
        # CHARACTERISTIC VARIABLES
        qp = u0[:, :, 2:] + c*u0[:, :, :2]
        qm = u0[:, :, 2:] - c*u0[:, :, :2]
        if  qp.max() > 1e1:
            return rho[:, :t], v[:, :t], tt
        u2s = torch.zeros((batch_size, nx, 4), device=rho.device)
        ff1 = torch.zeros((batch_size, nx, 4), device=rho.device)

        ff1[:, :, 0] = (-(recon_vect_modified(qp[:, :, 0], qm[:, :, 0], 2, c1, epsilon) - recon_vect_modified(torch.roll(qp[:, :,0], 1, -1), torch.roll(qm[:, :, 0], 1, -1), 2, c1, epsilon))/delta_x)
        ff1[:, :, 1] = (-(recon_vect_modified(qp[:, :, 1], qm[:, :, 1], 3, c2, epsilon) - recon_vect_modified(torch.roll(qp[:, :, 1], 1, -1), torch.roll(qm[:, :,1], 1, -1), 3, c2, epsilon))/delta_x)
        ff1[:, :, 2] = -c1.unsqueeze(-1)**2*((recon_vect_modified(qp[:, :, 0], qm[:, :, 0], 0, c1, epsilon) - recon_vect_modified(torch.roll(qp[:, :, 0], 1, -1), torch.roll(qm[:, :, 0], 1, -1), 0, c1, epsilon))/delta_x)
        ff1[:, :, 3] = -c2.unsqueeze(-1)**2*((recon_vect_modified(qp[:, :, 1], qm[:, :, 1], 1, c2, epsilon) - recon_vect_modified(torch.roll(qp[:, :, 1], 1, -1), torch.roll(qm[:, :, 1], 1, -1), 1, c2, epsilon))/delta_x)

        u2s = u0 + delta_t * ab21 * ff1
        u2 = u2s
        u2[:, :, 2] = (u2s[:, :, 2] + edt*a22*flx_modified(0, u2[:, :, 0], u2[:, :, 1], tol))/(1. + edt*a22)
        u2[:, :, 3] = (u2s[:, :, 3] + edt*a22*flx_modified(1, u2[:, :, 0], u2[:, :, 1], tol))/(1. + edt*a22)

        u2[:, :, 0] = torch.where(u2[:, :, 0] <= tol, 0, u2[:, :, 0])
        u2[:, :, 1] = torch.where(u2[:, :, 0] <= tol, 0, u2[:, :, 1])

        u2[:, :2, :2] = u2[:, [2], :2]
        u2[:, -2:, :2] = u2[:, [-3], :2] 

        for k in [2, 3]:
            u2[:, :2, k] = flx_modified(k-2, u2[:, :2, 0], u2[:, :2, 1], tol)
            u2[:, -2:, k] = flx_modified(k-2, u2[:, -2:, 0], u2[:, -2:, 1], tol)

        u2, v_current = u_cal_modified(u2, v_current, tol)

        qp2 = u2[:, :, 2:] + c*u2[:, :, :2]
        qm2 = u2[:, :, 2:] - c*u2[:, :, :2]


        ff1 = torch.zeros((batch_size, nx, 4), device=rho.device)
        ff2 = torch.zeros((batch_size, nx, 4), device=rho.device)

        ff1[:, :, 0] = (-(recon_vect_modified(qp[:, :, 0], qm[:, :, 0], 2, c1, epsilon) - recon_vect_modified(torch.roll(qp[:, :, 0], 1, -1), torch.roll(qm[:, :, 0], 1, -1), 2, c1, epsilon))/delta_x)
        ff1[:, :, 1] = (-(recon_vect_modified(qp[:, :, 1], qm[:, :, 1], 3, c2, epsilon) - recon_vect_modified(torch.roll(qp[:, :, 1], 1, -1), torch.roll(qm[:, :, 1], 1, -1), 3, c2, epsilon))/delta_x)
        ff1[:, :, 2] = -c1.unsqueeze(-1)**2*((recon_vect_modified(qp[:, :, 0], qm[:, :, 0], 0, c1, epsilon) - recon_vect_modified(torch.roll(qp[:, :, 0], 1, -1), torch.roll(qm[:, :, 0], 1, -1), 0, c1, epsilon))/delta_x)
        ff1[:, :, 3] = -c2.unsqueeze(-1)**2*((recon_vect_modified(qp[:, :, 1], qm[:, :, 1], 1, c2, epsilon) - recon_vect_modified(torch.roll(qp[:, :, 1], 1, -1), torch.roll(qm[:, :, 1], 1, -1), 1, c2, epsilon))/delta_x)

        ff2[:, :, 0] = (-(recon_vect_modified(qp2[:, :, 0], qm2[:, :, 0], 2, c1, epsilon) - recon_vect_modified(torch.roll(qp2[:, :, 0], 1, -1), torch.roll(qm2[:, :, 0], 1, -1), 2, c1, epsilon))/delta_x)
        ff2[:, :, 1] = (-(recon_vect_modified(qp2[:, :, 1], qm2[:, :, 1], 3, c2, epsilon) - recon_vect_modified(torch.roll(qp2[:, :, 1], 1, -1), torch.roll(qm2[:, :, 1], 1, -1), 3, c2, epsilon))/delta_x)
        ff2[:, :, 2] = -c1.unsqueeze(-1)**2*((recon_vect_modified(qp2[:, :, 0], qm2[:, :, 0], 0, c1, epsilon) - recon_vect_modified(torch.roll(qp2[:, :, 0], 1, -1), torch.roll(qm2[:, :, 0], 1, -1), 0, c1, epsilon))/delta_x)
        ff2[:, :, 3] = -c2.unsqueeze(-1)**2*((recon_vect_modified(qp2[:, :, 1], qm2[:, :, 1], 1, c2, epsilon) - recon_vect_modified(torch.roll(qp2[:, :, 1], 1, -1), torch.roll(qm2[:, :, 1], 1, -1), 1, c2, epsilon))/delta_x)

        u3s = torch.zeros((batch_size, nx, 4), device=rho.device)
        u3s = u0 + delta_t*ab31*ff1 + delta_t*ab32*ff2

        u3 = u3s
        for k in [2, 3]:
            u3[:, :, k] = (u3s[:, :, k] - edt*a32*(u2[:, :, k] - flx_modified(k-2, u2[:, :, 0], u2[:, :, 1], tol)) + edt*a33*flx_modified(k-2, u3[:, :, 0], u3[:, :, 1], tol))/(1. + edt*a33)

        u3[:, :, 0] = torch.where(u3[:, :, 0] <= tol, 0, u3[:, :, 0])
        u3[:, :, 1] = torch.where(u3[:, :, 0] <= tol, 0, u3[:, :, 1])
            
        # Bountary Values for l=3 RK stage 

        # Neuman
        u3[:, :2, :2] = u3[:, [2], :2]
        u3[:, -2:, :2] = u3[:, [-2], :2] 

        for k in [2, 3]:
            u3[:, :2, k] = flx_modified(k-2, u3[:, :2, 0], u3[:, :2, 1], tol)
            u3[:, -2:, k] = flx_modified(k-2, u3[:, -2:, 0], u3[:, -2:, 1], tol)

        u3, v_current = u_cal_modified(u3, v_current, tol)

        qp3 = u3[:, :, 2:] + c*u3[:, :, :2]
        qm3 = u3[:,:, 2:] - c*u3[:, :, :2]

        ff2 = torch.zeros((batch_size, nx, 4), device=rho.device)
        ff3 = torch.zeros((batch_size, nx, 4), device=rho.device)

        ff2[:, :, 0] = (-(recon_vect_modified(qp2[:, :, 0], qm2[:, :, 0], 2, c1, epsilon) - recon_vect_modified(torch.roll(qp2[:, :,0], 1, -1), torch.roll(qm2[:, :,0], 1, -1), 2, c1, epsilon))/delta_x)
        ff2[:, :, 1] = (-(recon_vect_modified(qp2[:, :, 1], qm2[:, :, 1], 3, c2, epsilon) - recon_vect_modified(torch.roll(qp2[:, :,1], 1, -1), torch.roll(qm2[:, :,1], 1, -1), 3, c2, epsilon))/delta_x)
        ff2[:, :, 2] = -c1.unsqueeze(-1)**2*((recon_vect_modified(qp2[:, :, 0], qm2[:, :, 0], 0, c1, epsilon) - recon_vect_modified(torch.roll(qp2[:, :,0], 1, -1), torch.roll(qm2[:, :,0], 1, -1), 0, c1, epsilon))/delta_x)
        ff2[:, :, 3] = -c2.unsqueeze(-1)**2*((recon_vect_modified(qp2[:, :, 1], qm2[:, :, 1], 1, c2, epsilon) - recon_vect_modified(torch.roll(qp2[:, :,1], 1, -1), torch.roll(qm2[:, :,1], 1, -1), 1, c2, epsilon))/delta_x)

        ff3[:, :, 0] = (-(recon_vect_modified(qp3[:, :, 0], qm3[:, :, 0], 2, c1, epsilon) - recon_vect_modified(torch.roll(qp3[:, :,0], 1, -1), torch.roll(qm3[:, :,0], 1, -1), 2, c1, epsilon))/delta_x)
        ff3[:, :, 1] = (-(recon_vect_modified(qp3[:, :, 1], qm3[:, :, 1], 3, c2, epsilon) - recon_vect_modified(torch.roll(qp3[:, :,1], 1, -1), torch.roll(qm3[:, :,1], 1, -1), 3, c2, epsilon))/delta_x)
        ff3[:, :, 2] = -c1.unsqueeze(-1)**2*((recon_vect_modified(qp3[:, :, 0], qm3[:, :, 0], 0, c1, epsilon) - recon_vect_modified(torch.roll(qp3[:, :,0], 1, -1), torch.roll(qm3[:, :,0], 1, -1), 0, c1, epsilon))/delta_x)
        ff3[:, :, 3] = -c2.unsqueeze(-1)**2*((recon_vect_modified(qp3[:, :, 1], qm3[:, :, 1], 1, c2, epsilon) - recon_vect_modified(torch.roll(qp3[:, :,1], 1, -1), torch.roll(qm3[:, :,1], 1, -1), 1, c2, epsilon))/delta_x)

        u0[:, :, :2] += delta_t*(ff2[:, :, :2] + ff3[:, :, :2])/2.
        for k in [2, 3]:
            u0[:, :, k] += 0.5*delta_t*(ff2[:, :, k] + ff3[:, :, k]) - (
                0.5*edt*(u2[:, :, k] - flx_modified(k-2, u2[:, :, 0], u2[:, :, 1], tol)) 
                + 0.5*edt*(u3[:, :, k] - flx_modified(k-2, u3[:, :, 0], u3[:, :, 1], tol)))

        u0[:, :, 0] = torch.where(u0[:, :, 0] <= tol, 0, u0[:, :, 0])
        u0[:, :, 1] = torch.where(u0[:, :, 0] <= tol, 0, u0[:, :, 1])


        # Bountary Values for time evolved at (n+1) solution values
        u0[:, :2, :2] = u0[:, [2], :2]
        u0[:, -2:, :2] = u0[:, [-3], :2] # 2.*u0[-3, :2] - u0[-3, :2] because -> U0(K,NPTS+1)= U0(K,NPTS) !2.d0*U0(K,NPTS)-U0(K,NPTS)
    
        for k in [2, 3]:
            u0[:, :2, k] = flx_modified(k-2, u0[:, :2, 0], u0[:, :2, 1], tol)
            u0[:, -2:, k] = flx_modified(k-2, u0[:, -2:, 0], u0[:, -2:, 1], tol)     

        u0, v_current = u_cal_modified(u0, v_current, tol)
                # Bountary Values for time evolved at (n+1) solution values
        u0[:, :2, :2] = u0[:, [2], :2]
        u0[:, -2:, :2] = u0[:, [-3], :2] # 2.*u0[-3, :2] - u0[-3, :2] because -> U0(K,NPTS+1)= U0(K,NPTS) !2.d0*U0(K,NPTS)-U0(K,NPTS)
    
        for k in [2, 3]:
            u0[:, :2, k] = flx_modified(k-2, u0[:, :2, 0], u0[:, :2, 1], tol)
            u0[:, -2:, k] = flx_modified(k-2, u0[:, -2:, 0], u0[:, -2:, 1], tol)     


        t+=1
        # rho[:, t] = u0[:, 0]
        v[:, :, t] = v_current

        time += delta_t

        qp = u0[:, :, 2:] + c*u0[:, :, :2]
        qm = u0[:, :, 2:] - c*u0[:, :, :2]

        # rrecp = torch.zeros(batch_size, nx, device=rho.device)
        # urecp = torch.zeros(batch_size, nx, device=rho.device)
        # for i in range(2, nx-3):
        #     rrecp[i] = recon(qp[:, :, 0], qm[:, :, 0], i, 0, c1, epsilon)
        #     urecp[i] = recon(qp[:, :, 1], qm[:, :, 1], i, 1, c2, epsilon)

        delta_t, c1, c2 = timestep_modified(v_current, u0, delta_x, cfl=.2, crc1=5e-3, crc2=1.e-3)
        if dt is not None:
            delta_t = dt
        c = torch.stack([c1, c2]).to(rho.device).T.unsqueeze(1)
        edt = delta_t / epsilon 
        tt.append(time)

        rho[:, :, t] = u0[:, :, 0]
        v[:, :, t] = v_current

        print(f'{time:.3e} {t:1f}')
        if time >= max_time or t == rho.shape[2]-1:
            # print(rho[:, t], v[:, t])
            return rho[:, :, :t+1], v[:, :, :t+1], tt
