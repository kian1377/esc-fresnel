from .math_module import xp, xcipy, ensure_np_array
from esc_fresnel import utils
from esc_fresnel.imshows import imshow1, imshow2, imshow3

import numpy as np
import astropy.units as u
import time
import copy
from IPython.display import display, clear_output
import matplotlib.pyplot as plt

def compute_jacobian(
        M, 
        calib_modes,
        control_mask, 
        amp=1e-9,
        channel=3,
        plot=False,
    ):

    Nmask = int(control_mask.sum())
    jac = xp.zeros((2*Nmask, M.Nacts))

    start = time.time()
    poke_command = xp.zeros((M.Nact, M.Nact))
    for i in range(M.Nacts):
        poke = xp.zeros(M.Nacts)
        poke[i] = amp
        poke_command[M.dm_mask] = poke 

        M.set_dm(poke_command, channel=channel)
        E_pos = M.calc_wf()

        M.set_dm(-poke_command, channel=channel)
        E_neg = M.calc_wf()

        response = ( E_pos - E_neg ) / (2 * amp)

        jac[::2, i] = response.real[control_mask]
        jac[1::2, i] = response.imag[control_mask]
        print(f"\tCalibrated mode {i+1:d}/{M.Nacts:d} in {time.time()-start:.3f}s", end='')
        print("\r", end="")

    M.reset_dm()
    # if plot:
    #     dm_response_map = xp.sqrt(xp.mean(xp.square( response_matrix.dot(calibration_modes.reshape(Nmodes, -1))), axis=0))
    #     dm_response_map = dm_response_map.reshape(I.Nact,I.Nact) / xp.max(dm_response_map)

    return jac

def run(M, 
        control_matrix,
        control_mask,
        dm_mask,
        data,
        channel=3,
        num_iterations=3,
        gain=0.5, 
        plot=False,
        vmin=1e-10,
    ):
    
    starting_itr = len(data['images'])
    total_command = copy.copy(data['commands'][-1]) if len(data['commands'])>0 else xp.zeros((M.Nact,M.Nact))

    del_command = xp.zeros((M.Nact,M.Nact)) # array to fill with actuator solutions
    Nacts = control_matrix.shape[0]
    Nmask = int(control_mask.sum())
    E_ab_vec = xp.zeros(2*Nmask)
    for i in range(num_iterations):
        
        E_ab = M.calc_wf()

        E_ab_vec[::2] = E_ab[control_mask].real
        E_ab_vec[1::2] = E_ab[control_mask].imag
        del_acts = - gain * control_matrix.dot(E_ab_vec)
        del_command[dm_mask] = del_acts[:Nacts]
        total_command += del_command
        M.set_dm(total_command, channel=channel)

        image_ni = M.snap()
        mean_ni = xp.mean(image_ni[control_mask])

        data['images'].append(copy.copy(image_ni))
        data['contrasts'].append(copy.copy(mean_ni))
        data['efields'].append(copy.copy(E_ab))
        data['commands'].append(copy.copy(total_command))
        data['del_commands'].append(copy.copy(del_command))

        if plot:
            imshow3(
                del_command, total_command, image_ni, 
                f'$\delta$DM Command', 
                f'Total Command', 
                f'Iteration {starting_itr + i:d} Image\nMean NI = {mean_ni:.3e}',
                cmap1='viridis', cmap2='viridis', 
                pxscl3=M.camsci_pxscl_lamDc, 
                lognorm3=True, 
                vmin3=vmin,
            )

    return data

