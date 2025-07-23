from .math_module import xp, xcipy, ensure_np_array
from esc_fresnel import utils, dm, props

import numpy as np
import astropy.units as u
from astropy.io import fits
import os
from pathlib import Path
import time
import copy

import poppy
from scipy.signal import windows

import matplotlib.pyplot as plt
plt.rcParams['image.origin'] = 'lower'
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm, Normalize, CenteredNorm
from matplotlib.patches import Circle, Rectangle
from IPython.display import display, clear_output

def acts_to_command(acts, dm_mask):
    Nact = dm_mask.shape[0]
    command = xp.zeros((Nact,Nact))
    command[dm_mask] = xp.array(acts)
    return command

class MODEL():
    def __init__(
            self, 
            dm_beam_diam=9.351e-3,
            lyot_pupil_diam=4.153e-3,
            lyot_stop_diam=3.7e-3,
            vortex_sign=1, 
            dm_ref=None, 
        ):

        # initialize physical parameters
        self.wavelength_c = 650e-9
        self.waves = np.array([self.wavelength_c])
        
        self.dm_beam_diam = dm_beam_diam
        self.lyot_pupil_diam = lyot_pupil_diam
        self.lyot_stop_diam = lyot_stop_diam
        self.lyot_ratio = (self.lyot_stop_diam / self.lyot_pupil_diam)
        self.control_rad = 34/2 * self.dm_beam_diam/10.2 * self.lyot_ratio
        self.psf_pixelscale = 3.76e-6
        self.psf_pixelscale_lamDc = 0.25
        self.npsf = 256

        self.Imax_ref = 1

        self.exit_pupil_prop_distance = None

        # initialize sampling parameters and load masks
        self.npix = 1000
        self.oversample = 2.048
        self.N = int(self.npix*self.oversample)

        self.dm_pxscl = self.dm_beam_diam / self.npix
        self.lyot_pxscl = self.lyot_pupil_diam / self.npix

        self.flip_dm = False
        self.reverse_lyot = False
        self.flip_lyot = False

        dm_wf = poppy.FresnelWavefront(beam_radius=self.dm_beam_diam/2*u.m, npix=self.npix, oversample=1) # pupil wavefront
        self.APERTURE = poppy.CircularAperture(radius=self.dm_beam_diam/2*u.m).get_transmission(dm_wf)
        self.APMASK = self.APERTURE>0

        self.AMP = xp.ones((self.npix,self.npix))
        self.OPD = xp.zeros((self.npix,self.npix))

        lyot_wf = poppy.FresnelWavefront(beam_radius=self.lyot_pupil_diam/2*u.m, npix=self.npix, oversample=1) # pupil wavefront
        self.LYOT = poppy.CircularAperture(radius=self.lyot_ratio*self.lyot_pupil_diam/2*u.m).get_transmission(lyot_wf)

        self.Nact = 34
        self.dm_shape = (self.Nact, self.Nact)
        self.act_spacing = 300e-6
        self.inf_sampling = self.act_spacing/self.dm_pxscl
        self.inf_fun = utils.make_gaussian_inf_fun(
            act_spacing=self.act_spacing,  
            sampling=self.inf_sampling, 
            coupling=0.15, 
            Nact=self.Nact+2,
        )
        self.Nsurf = self.inf_fun.shape[0]

        self.dm_ref = copy.copy(dm_ref) if dm_ref is not None else xp.zeros((self.Nact, self.Nact))
        self.dm_channels = xp.zeros((10, self.Nact, self.Nact))
        self.dm_channels[0] = self.dm_ref
        self.dm_total = xp.sum(self.dm_channels, axis=0)

        y,x = (xp.indices((self.Nact, self.Nact)) - self.Nact//2 + 1/2)
        r = xp.sqrt(x**2 + y**2)
        self.dm_mask = r<(self.Nact/2 + 1/2)
        self.Nacts = int(self.dm_mask.sum())

        self.inf_fun_fft = xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift(self.inf_fun,)))
        # DM command coordinates
        xc = self.inf_sampling*(xp.linspace(-self.Nact//2, self.Nact//2-1, self.Nact) + 1/2)
        yc = self.inf_sampling*(xp.linspace(-self.Nact//2, self.Nact//2-1, self.Nact) + 1/2)

        # Influence function frequncy sampling
        fx = xp.fft.fftshift(xp.fft.fftfreq(self.Nsurf))
        fy = xp.fft.fftshift(xp.fft.fftfreq(self.Nsurf))

        # forward DM model MFT matrices
        self.Mx = xp.exp(-1j*2*np.pi*xp.outer(fx,xc))
        self.My = xp.exp(-1j*2*np.pi*xp.outer(yc,fy))

        self.Mx_back = xp.exp(1j*2*np.pi*xp.outer(xc,fx))
        self.My_back = xp.exp(1j*2*np.pi*xp.outer(fy,yc))

        # Vortex model parameters
        self.oversample_vortex = 4.096
        self.N_vortex_lres = int(self.npix*self.oversample_vortex)
        self.lres_sampling = 1/self.oversample_vortex # low resolution sampling in lam/D per pixel
        self.lres_win_size = int(30/self.lres_sampling)
        w1d = xp.array(windows.tukey(self.lres_win_size, 1, False))
        self.lres_window = utils.pad_or_crop(xp.outer(w1d, w1d), self.N_vortex_lres)
        self.vortex_lres = props.make_vortex_phase_mask(self.N_vortex_lres, sign=vortex_sign)

        self.hres_sampling = 0.025 # lam/D per pixel; this value is chosen empirically
        self.N_vortex_hres = int(np.round(30/self.hres_sampling))
        self.hres_win_size = int(30/self.hres_sampling)
        w1d = xp.array(windows.tukey(self.hres_win_size, 1, False))
        self.hres_window = utils.pad_or_crop(xp.outer(w1d, w1d), self.N_vortex_hres)
        self.vortex_hres = props.make_vortex_phase_mask(self.N_vortex_hres, sign=vortex_sign)

        y,x = (xp.indices((self.N_vortex_hres, self.N_vortex_hres)) - self.N_vortex_hres//2)*self.hres_sampling
        r = xp.sqrt(x**2 + y**2)
        self.hres_dot_mask = r>=0.15

        self.use_vortex = True
        self.dm_command = xp.zeros((self.Nact, self.Nact))
    
    def forward(
            self, 
            actuators, 
            wavelength=650e-9, 
            use_vortex=True, 
            return_ints=False, 
            plot=False,
            fancy_plot=False, 
            fancy_plot_fname=None,
        ):

        A = xp.zeros((self.Nact,self.Nact))
        A[self.dm_mask] = xp.array(actuators)
        A_hat = self.Mx@A@self.My
        S_DM_hat = self.inf_fun_fft * A_hat
        S_DM = xp.fft.fftshift(xp.fft.ifft2(xp.fft.ifftshift(S_DM_hat,))).real
        phi_DM = 4*xp.pi/wavelength * utils.pad_or_crop(S_DM, self.N)
        DM_PHASOR = xp.exp(1j * phi_DM)
        if self.flip_dm: DM_PHASOR = xp.rot90(xp.rot90(DM_PHASOR))

        # Initialize the wavefront
        WFE =  utils.pad_or_crop(self.AMP, self.N) * xp.exp(1j * 2*xp.pi/wavelength * utils.pad_or_crop(self.OPD, self.N))
        E_EP = utils.pad_or_crop(self.APERTURE.astype(xp.complex128), self.N) * WFE / xp.sqrt(self.Imax_ref)
        if plot: utils.imshow([xp.abs(E_EP), xp.angle(E_EP)], titles=['EP WF'], npix=2*[1.5*self.npix], cmaps=['plasma','twilight'])

        E_DM = E_EP * utils.pad_or_crop(DM_PHASOR, self.N)
        if plot: utils.imshow([xp.abs(E_DM), xp.angle(E_DM)], titles=['EP WF'], npix=2*[1.5*self.npix], cmaps=['plasma','twilight'])

        if use_vortex:
            lres_wf = utils.pad_or_crop(E_DM, self.N_vortex_lres) # pad to the larger array for the low res propagation
            fp_wf_lres = props.fft(lres_wf)
            fp_wf_lres *= self.vortex_lres * (1 - self.lres_window) # apply low res FPM and inverse Tukey window
            pupil_wf_lres = props.ifft(fp_wf_lres)
            pupil_wf_lres = utils.pad_or_crop(pupil_wf_lres, self.N)
            if plot: utils.imshow([xp.abs(pupil_wf_lres), xp.angle(pupil_wf_lres)], titles=['FFT Lyot WF'], npix=2*[1.5*self.npix], cmaps=['plasma','twilight'])

            fp_wf_hres = props.mft_forward(
                E_DM, 
                self.npix, 
                self.N_vortex_hres, 
                self.hres_sampling, 
                convention='-',
                fp_centering='odd',
                pp_centering='odd',
            )
            fp_wf_hres *= self.vortex_hres * self.hres_window * self.hres_dot_mask # apply high res FPM, window, and dot mask
            pupil_wf_hres = props.mft_reverse(
                fp_wf_hres, 
                self.hres_sampling, 
                self.npix, 
                self.N, 
                convention='+',
                fp_centering='odd',
                pp_centering='odd',
            )
            if plot: utils.imshow([xp.abs(pupil_wf_hres), xp.angle(pupil_wf_hres)], titles=['MFT Lyot WF'], npix=2*[1.5*self.npix], cmaps=['plasma','twilight'])

            E_LP = (pupil_wf_lres + pupil_wf_hres)
            if plot: utils.imshow([xp.abs(E_LP), xp.angle(E_LP)], titles=['Total Lyot WF'], npix=2*[1.5*self.npix], cmaps=['plasma','twilight'])
        else:
            E_LP = E_DM

        if self.reverse_lyot: E_LP = xp.rot90(xp.rot90(E_LP))
        if self.flip_lyot: E_LP = xp.fliplr(E_LP)

        E_LS = utils.pad_or_crop(self.LYOT, self.N) * E_LP
        if plot: utils.imshow(xp.abs(E_LS), xp.angle(E_LS), titles=['After Lyot WF'], npix=2*[1.5*self.npix], cmaps=['plasma','twilight'])
        
        if self.exit_pupil_prop_distance is not None:
            E_LS = props.ang_spec(E_LS, wavelength, self.exit_pupil_prop_distance, self.lyot_pxscl)
            if plot: utils.imshow(xp.abs(E_LS), xp.angle(E_LS), titles=['Lyot WF after propagation'], npix=2*[1.5*self.npix], cmaps=['plasma','twilight'])

        psf_pixelscale_lamD = self.psf_pixelscale_lamDc * self.wavelength_c/wavelength
        E_FP = props.mft_forward(E_LS, self.npix * self.lyot_ratio, self.npsf, psf_pixelscale_lamD)
        if plot: utils.imshow([xp.abs(E_FP)**2, xp.angle(E_FP)], titles=['At Camera WF'], cmaps=['magma','twilight'], norms=[LogNorm()])

        if fancy_plot: 
            fancy_plot_forward(
                A, E_EP, DM_PHASOR, E_LP, E_FP, 
                self,
                fname=fancy_plot_fname,
            )

        if return_ints:
            return E_FP, E_EP, DM_PHASOR, E_DM, E_LP, E_LS
        else:
            return E_FP
        
    def getattr(self, attr):
        return getattr(self, attr)
    
    def setattr(self, attr, val):
        setattr(self, attr, val)
    
    def zero_dm(self, channel=1):
        self.dm_channels[channel] = xp.zeros((34,34))
        self.dm_total = xp.sum(self.dm_channels, axis=0)

    def reset_dm(self):
        self.dm_channels = xp.zeros((10,34,34))
        self.dm_channels[0] = self.dm_ref
        self.dm_total = xp.sum(self.dm_channels, axis=0)

    def set_dm(self, command, channel=1):
        self.dm_channels[channel] = copy.copy(command)
        self.dm_total = xp.sum(self.dm_channels, axis=0)

    def add_dm(self, command, channel=1):
        old = self.dm_channels[channel]
        self.dm_channels[channel] = copy.copy(old + command)
        self.dm_total = xp.sum(self.dm_channels, axis=0)

    def get_dm(self, channel=1):
        return copy.copy(self.dm_channels[channel])

    def get_dm_total(self):
        return self.dm_total

    def calc_pupil(self, ):
        actuators = self.dm_total[self.dm_mask]

        A = xp.zeros((self.Nact,self.Nact))
        A[self.dm_mask] = xp.array(actuators)
        A_hat = self.Mx@A@self.My
        S_DM_hat = self.inf_fun_fft * A_hat
        S_DM = xp.fft.fftshift(xp.fft.ifft2(xp.fft.ifftshift(S_DM_hat,))).real
        phi_DM = 4*xp.pi/self.wavelength_c * utils.pad_or_crop(S_DM, self.N)
        DM_PHASOR = xp.exp(1j * phi_DM)
        if self.flip_dm: DM_PHASOR = xp.rot90(xp.rot90(DM_PHASOR))

        # Initialize the wavefront
        WFE =  utils.pad_or_crop(self.AMP, self.N) * xp.exp(1j * 2*xp.pi/self.wavelength_c * utils.pad_or_crop(self.OPD, self.N))
        E_EP = utils.pad_or_crop(self.APERTURE.astype(xp.complex128), self.N) * WFE / xp.sqrt(self.Imax_ref)
        E_DM = E_EP * utils.pad_or_crop(DM_PHASOR, self.N)

        EP_AMP = utils.pad_or_crop(xp.abs(E_DM), self.npix) * self.APERTURE
        EP_OPD = self.wavelength_c / (2*xp.pi) * utils.pad_or_crop(xp.angle(E_DM), self.npix) * self.APERTURE
        return EP_AMP, EP_OPD
        
        
    def calc_wf(self, wavelength=650e-9):
        actuators = self.dm_total[self.dm_mask]
        fpwf = self.forward(actuators, wavelength, use_vortex=self.use_vortex,)
        return fpwf
        
    def snap(self):
        Nwaves = len(self.waves)
        im = 0.0
        for i in range(Nwaves):
            actuators = self.dm_total[self.dm_mask]
            fpwf = self.forward(actuators, self.waves[i], use_vortex=self.use_vortex,)
            im += xp.abs( fpwf )**2 / Nwaves
        return im

def val_and_grad(
        del_acts, 
        M, 
        rmad_vars, 
        verbose=False, 
        plot=False, 
        fancy_plot=False, 
        fancy_plot_fname=None,
    ):
    # Convert array arguments into correct types
    del_acts = xp.array(del_acts)
    del_acts_waves = del_acts/M.wavelength_c

    current_acts = rmad_vars['current_acts']
    E_ab = rmad_vars['E_ab']
    E_FP_NOM = rmad_vars['E_FP_NOM']
    wavelength = rmad_vars['wavelength']
    control_mask = rmad_vars['control_mask']
    r_cond = rmad_vars['r_cond']

    E_ab_l2norm = E_ab[control_mask].dot(E_ab[control_mask].conjugate()).real

    # Compute E_DM using the forward DM model
    E_FP_with_delA, E_EP, DM_PHASOR, _, _, _ = M.forward(
        current_acts + del_acts, 
        wavelength, 
        use_vortex=True, 
        return_ints=True,
    )
    deltaE = E_FP_with_delA - E_FP_NOM

    # compute the cost function
    E_predicted = E_ab + deltaE # take the measured E-field and add the model-based deltaE from new actuator command
    E_predicted_vec = E_predicted[control_mask] # make sure to do array indexing
    J_delE = E_predicted_vec.dot(E_predicted_vec.conjugate()).real
    J_c = r_cond * del_acts_waves.dot(del_acts_waves)
    J = (J_delE + J_c) / E_ab_l2norm
    if verbose: 
        print(f'\tCost-function J_delE: {J_delE:.3f}')
        print(f'\tCost-function J_c: {J_c:.3f}')
        print(f'\tCost-function normalization factor: {E_ab_l2norm:.3f}')
        print(f'\tTotal cost-function value: {J:.3f}\n')

    # Compute the gradient with the adjoint model
    E_predicted_masked = control_mask * E_predicted # still a 2D array
    # E_predicted_masked = xcipy.ndimage.rotate(E_predicted_masked, -M.det_rotation, reshape=False, order=5)
    dJ_ddeltaE = 2 * E_predicted_masked / E_ab_l2norm
    if plot: utils.imshow([xp.abs(dJ_ddeltaE)**2, xp.angle(dJ_ddeltaE)], titles=['dJ_ddelta_E'], cmaps=['magma', 'twilight'], norms=[LogNorm(vmin=xp.max(xp.abs(dJ_ddeltaE)**2)/1e3)])

    psf_pixelscale_lamD = M.psf_pixelscale_lamDc * M.wavelength_c/wavelength
    dJ_dE_LS = props.mft_reverse(dJ_ddeltaE, psf_pixelscale_lamD, M.npix * M.lyot_ratio, M.N, convention='+')
    if plot: utils.imshow([xp.abs(dJ_dE_LS), xp.angle(dJ_dE_LS)], titles=['dJ_dE_LS'], npix=2*[1.5*M.npix], cmaps=['plasma','twilight'])
    
    if M.exit_pupil_prop_distance is not None:
        dJ_dE_LS = props.ang_spec(dJ_dE_LS, wavelength, -M.exit_pupil_prop_distance, M.lyot_pxscl)
        if plot: utils.imshow([xp.abs(dJ_dE_LS), xp.angle(dJ_dE_LS)], titles=['dJ_dE_LS'], npix=2*[1.5*M.npix], cmaps=['plasma','twilight'])

    dJ_dE_LP = dJ_dE_LS * utils.pad_or_crop(M.LYOT, M.N)
    if M.flip_lyot: dJ_dE_LP = xp.fliplr(dJ_dE_LP)
    if M.reverse_lyot: dJ_dE_LP = xp.rot90(xp.rot90(dJ_dE_LP))
    if plot: utils.imshow([xp.abs(dJ_dE_LP), xp.angle(dJ_dE_LP)], titles=['dJ_dE_LP'], npix=2*[1.5*M.npix], cmaps=['plasma','twilight'])

    # Now we have to split and back-propagate the gradient along the two branches used to model the vortex.
    # So one branch for the FFT vortex procedure and one for the MFT vortex procedure. 
    dJ_dE_LP_fft = utils.pad_or_crop(copy.copy(dJ_dE_LP), M.N_vortex_lres)
    dJ_dE_FPM_fft = props.fft(dJ_dE_LP_fft)
    dJ_dE_FP_fft = M.vortex_lres.conj() * (1 - M.lres_window) * dJ_dE_FPM_fft
    dJ_dE_PUP_fft = props.ifft(dJ_dE_FP_fft)
    dJ_dE_PUP_fft = utils.pad_or_crop(dJ_dE_PUP_fft, M.N)
    if plot: utils.imshow([xp.abs(dJ_dE_PUP_fft), xp.angle(dJ_dE_PUP_fft)],  titles=['dJ_dE_PUP_fft'], npix=2*[1.5*M.npix], cmaps=['plasma','twilight'])

    dJ_dE_LP_mft = utils.pad_or_crop(copy.copy(dJ_dE_LP), M.N)
    dJ_dE_FPM_mft = props.mft_forward(dJ_dE_LP_mft,  M.npix, M.N_vortex_hres, M.hres_sampling, convention='-')
    dJ_dE_FP_mft = M.vortex_hres.conj() * M.hres_window * M.hres_dot_mask * dJ_dE_FPM_mft
    dJ_dE_PUP_mft = props.mft_reverse(dJ_dE_FP_mft, M.hres_sampling, M.npix, M.N, convention='+')
    if plot: utils.imshow([xp.abs(dJ_dE_PUP_mft), xp.angle(dJ_dE_PUP_mft)], titles=['dJ_dE_PUP_mft'], npix=2*[1.5*M.npix], cmaps=['plasma','twilight'])

    dJ_dE_PUP = dJ_dE_PUP_fft + dJ_dE_PUP_mft
    if plot: utils.imshow([xp.abs(dJ_dE_PUP), xp.angle(dJ_dE_PUP)], titles=['dJ_dE_PUP'], npix=2*[1.5*M.npix], cmaps=['plasma','twilight'])

    dJ_dS_DM = 4*xp.pi / wavelength * xp.imag(dJ_dE_PUP * E_EP.conj() * DM_PHASOR.conj())
    if M.flip_dm: dJ_dS_DM = xp.rot90(xp.rot90(dJ_dS_DM))
    if plot: utils.imshow([xp.real(dJ_dS_DM), xp.imag(dJ_dS_DM)], titles=['dJ_dS_DM'], npix=2*[1.5*M.npix], cmaps=['viridis','viridis'])

    # Now pad back to the array size fo the DM surface to back propagate through the adjoint DM model
    dJ_dS_DM = utils.pad_or_crop(dJ_dS_DM, M.Nsurf)
    dJ_dS_DM_hat = xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift(dJ_dS_DM.real)))
    dJ_dA_hat = M.inf_fun_fft.conjugate() * dJ_dS_DM_hat
    dJ_dA = M.Mx_back@dJ_dA_hat@M.My_back / ( M.Nsurf * M.Nact * M.Nact ) # why I have to divide by this constant is beyond me
    if plot: utils.imshow([dJ_dA.real, dJ_dA.imag], titles=['dJ_dA'], npix=2*[1.5*M.npix], cmaps=['viridis','viridis'])
    
    dJ_dA = dJ_dA[M.dm_mask].real + xp.array( r_cond * 2*del_acts_waves )

    if fancy_plot: fancy_plot_adjoint(dJ_ddeltaE, dJ_dE_LP, dJ_dE_PUP, dJ_dS_DM, dJ_dA, control_mask, M, fname=fancy_plot_fname)
    
    return ensure_np_array(J), ensure_np_array(dJ_dA)

def val_and_grad_bb(
        del_acts, 
        M, 
        actuators, 
        E_abs, 
        control_mask, 
        waves, 
        r_cond, 
        weights=None, 
        verbose=False, 
        plot=False, 
        fancy_plot=False, 
    ):
    # del_acts, M, actuators, E_ab, control_mask, wavelength, r_cond,
    Nwaves = len(waves)
    E_abs = xp.array(E_abs)
    del_acts_waves = del_acts/M.wavelength_c

    r_cond_mono = 0
    J_monos = np.zeros(Nwaves)
    dJ_dA_monos = np.zeros((Nwaves, M.Nacts))
    for i in range(Nwaves):
        wavelength = waves[i]
        E_ab = E_abs[i]
        J_mono, dJ_dA_mono = val_and_grad(
            del_acts, 
            M, 
            actuators, 
            E_ab, 
            control_mask, 
            wavelength, 
            r_cond_mono, 
            verbose=verbose, 
            plot=plot, 
            fancy_plot=fancy_plot,
        )
        J_monos[i] = J_mono
        dJ_dA_monos[i] = dJ_dA_mono

    # imshows.imshow1(acts_to_command(dJ_dA_monos[2] - dJ_dA_monos[0], M.dm_mask))

    if weights is None: 
        weights = np.array(Nwaves*[1])
        # TODO: implement weights for each wavelength correctly

    J_bb = np.sum(J_monos)/Nwaves + r_cond * del_acts_waves.dot(del_acts_waves)
    dJ_dA_bb = np.sum(dJ_dA_monos, axis=0) + ensure_np_array( r_cond * 2*del_acts_waves )
    
    return J_bb, dJ_dA_bb

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm

def make_arr_extent(pxscl, shape):
    xlim = shape[1]/2 * pxscl
    ylim = shape[0]/2 * pxscl
    return [-xlim, xlim, -ylim, ylim]

def fancy_plot_forward(
        command, 
        E_EP, 
        DM_PHASOR, 
        E_LP, 
        E_FP, 
        M,
        wspace=0.3,
        hspace=-0.1,
        fname=None,
    ):
    S_DM = ensure_np_array(M.wavelength_c/(2*xp.pi) * utils.pad_or_crop(xp.angle(DM_PHASOR), 1.25*M.npix) )
    E_PUP = ensure_np_array(utils.pad_or_crop(E_EP * DM_PHASOR, 1.25*M.npix))
    E_LP = ensure_np_array(utils.pad_or_crop(E_LP, 1.25*M.npix))
    E_FP = ensure_np_array(E_FP)

    dm_extent = make_arr_extent(M.dm_pxscl*1e3, S_DM.shape)
    pup_extent = make_arr_extent(M.dm_pxscl*1e3, E_PUP.shape)
    lyot_extent = make_arr_extent(M.lyot_pxscl*1e3, E_LP.shape)
    fp_extent = make_arr_extent(M.psf_pixelscale_lamDc, E_FP.shape)

    fig = plt.figure(figsize=(20,10), dpi=125)
    gs = GridSpec(2, 5, figure=fig)

    title_fs = 16
    label_fs = 14

    ax = fig.add_subplot(gs[:, 0])
    im = ax.imshow(ensure_np_array(command), cmap='viridis', norm=CenteredNorm())
    ax.set_title('DM Command\n'+r'$A$', fontsize=title_fs)
    ax.set_xlabel('X [Actuators]', fontsize=label_fs)
    ax.set_ylabel('Y [Actuators]', fontsize=label_fs)
    ax.set_xticks(np.arange(0, 35, 5))
    ax.set_yticks(ax.get_xticks())
    # ax.set_yticks(np.arange(0, 35, 5))
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="4%", pad=0.075)
    # cbar = fig.colorbar(im, cax=cax)
    # cbar.ax.set_ylabel('nm', rotation=0, labelpad=5)

    print(dm_extent)
    ax = fig.add_subplot(gs[:, 1])
    im = ax.imshow(S_DM, cmap='viridis', norm=CenteredNorm(), extent=dm_extent)
    ax.set_title('DM Surface\n'+r'$S_{DM}$', fontsize=title_fs)
    ax.set_xlabel('X [mm]', fontsize=label_fs)
    ax.set_ylabel('Y [mm]', fontsize=label_fs, labelpad=0)
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="4%", pad=0.075)
    # cbar = fig.colorbar(im, cax=cax)
    # cbar.ax.set_ylabel('nm', rotation=0, labelpad=5)

    ax = fig.add_subplot(gs[0, 2])
    im = ax.imshow(np.abs(E_PUP), cmap='plasma', extent=pup_extent)
    ax.set_title('Total Pupil Amplitude\n'+r'$|E_{PUP}|$', fontsize=title_fs)
    # ax.set_xlabel('X [mm]', fontsize=label_fs)
    ax.set_ylabel('Y [mm]', fontsize=label_fs, labelpad=0)
    # ax.set_xticks([])
    # ax.set_yticks([])
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="4%", pad=0.075)
    # cbar = fig.colorbar(im, cax=cax)
    # cbar.ax.set_ylabel('', rotation=0, labelpad=5)

    ax = fig.add_subplot(gs[1, 2])
    im = ax.imshow(np.angle(E_PUP), cmap='twilight', extent=pup_extent)
    ax.set_title('Total Pupil Phase\n'+r'$\angle E_{PUP}$', fontsize=title_fs)
    ax.set_xlabel('X [mm]', fontsize=label_fs)
    ax.set_ylabel('Y [mm]', fontsize=label_fs, labelpad=0)
    # ax.set_xticks([])
    # ax.set_yticks([])
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="4%", pad=0.075)
    # cbar = fig.colorbar(im, cax=cax)
    # cbar.ax.set_ylabel('', rotation=0, labelpad=5)

    ax = fig.add_subplot(gs[0, 3])
    im = ax.imshow(np.abs(E_LP), cmap='plasma', extent=lyot_extent)
    ax.set_title('Lyot Pupil Amplitude\n'+r'$|E_{LP}|$', fontsize=title_fs)
    # ax.set_xlabel('X [mm]', fontsize=label_fs)
    ax.set_ylabel('Y [mm]', fontsize=label_fs, labelpad=0)
    # ax.set_xticks([])
    # ax.set_yticks([])
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="4%", pad=0.075)
    # cbar = fig.colorbar(im, cax=cax)
    # cbar.ax.set_ylabel('', rotation=0, labelpad=5)

    ax = fig.add_subplot(gs[1, 3])
    im = ax.imshow(np.angle(E_LP), cmap='twilight',  extent=lyot_extent)
    ax.set_title('Lyot Pupil Phase\n'+r'$\angle E_{LP}$', fontsize=title_fs)
    ax.set_xlabel('X [mm]', fontsize=label_fs)
    ax.set_ylabel('Y [mm]', fontsize=label_fs, labelpad=0)
    # ax.set_xticks([])
    # ax.set_yticks([])
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="4%", pad=0.075)
    # cbar = fig.colorbar(im, cax=cax)
    # cbar.ax.set_ylabel('', rotation=0, labelpad=5)

    ax = fig.add_subplot(gs[0, 4])
    im = ax.imshow(np.abs(E_FP)**2, cmap='magma', norm=LogNorm(vmin=1e-7, vmax=1e-3),  extent=fp_extent)
    ax.set_title('Focal Plane Intensity\n'+r'$|E_{FP}|^2$', fontsize=title_fs)
    # ax.set_xlabel('X [$\lambda/D$]', fontsize=label_fs)
    ax.set_ylabel('Y [$\lambda/D$]', fontsize=label_fs, labelpad=-5)
    # ax.set_xticks([])
    # ax.set_yticks([])
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="4%", pad=0.075)
    # cbar = fig.colorbar(im, cax=cax)
    # cbar.ax.set_ylabel('', rotation=0, labelpad=5)

    ax = fig.add_subplot(gs[1, 4])
    im = ax.imshow(np.angle(E_FP), cmap='twilight',  extent=fp_extent)
    ax.set_title('Focal Plane Phase\n'+r'$\angle E_{FP}$', fontsize=title_fs)
    ax.set_xlabel('X [$\lambda/D$]', fontsize=label_fs)
    ax.set_ylabel('Y [$\lambda/D$]', fontsize=label_fs, labelpad=-5)
    # ax.set_xticks([])
    # ax.set_yticks([])
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="4%", pad=0.075)
    # cbar = fig.colorbar(im, cax=cax)
    # cbar.ax.set_ylabel('', rotation=0, labelpad=5)

    plt.subplots_adjust(
        wspace=wspace,
        hspace=hspace,
    )
    if fname is not None: fig.savefig(fname, format='pdf', bbox_inches="tight")

def fancy_plot_adjoint(
        dJ_dE_delA, 
        dJ_dE_LP, 
        dJ_dE_PUP, 
        dJ_dS_DM, 
        dJ_dA, 
        control_mask, 
        M,
        wspace=0.3,
        hspace=0.2,
        fname=None,
    ):

    control_mask = ensure_np_array(utils.pad_or_crop(control_mask, 100) )
    dJ_dE_delA = ensure_np_array(utils.pad_or_crop(dJ_dE_delA, 100))
    dJ_dE_LP = ensure_np_array(utils.pad_or_crop(dJ_dE_LP, 1.25*M.npix))
    dJ_dE_PUP = ensure_np_array(utils.pad_or_crop(dJ_dE_PUP, 1.25*M.npix))
    dJ_dS_DM = ensure_np_array(utils.pad_or_crop(dJ_dS_DM, int(1.25*M.npix)))
    dm_grad = ensure_np_array(acts_to_command(dJ_dA, M.dm_mask))

    fig = plt.figure(figsize=(20,10), dpi=125)
    gs = GridSpec(2, 5, figure=fig)
    
    dm_extent = make_arr_extent(M.dm_pxscl*1e3, dJ_dS_DM.shape)
    pup_extent = make_arr_extent(M.dm_pxscl*1e3, dJ_dE_PUP.shape)
    lyot_extent = make_arr_extent(M.lyot_pxscl*1e3, dJ_dE_LP.shape)
    fp_extent = make_arr_extent(M.psf_pixelscale_lamDc, dJ_dE_delA.shape)

    title_fs = 18
    label_fs = 14

    ax = fig.add_subplot(gs[0, 0])
    # ax.imshow(np.abs(dJ_dE_DM)**2, cmap='magma', norm=LogNorm(vmin=1e-6))
    ax.imshow(np.abs(dJ_dE_delA)**2 * control_mask, cmap='magma', norm=LogNorm(vmin=1e-6), extent=fp_extent)
    ax.set_title('Intensity of Gradient\nat Focal Plane\n' + r'$| \frac{\partial J}{\partial \delta E} |^2$', fontsize=title_fs)
    # ax.set_xlabel('X [$\lambda/D$]', fontsize=label_fs)
    ax.set_ylabel('Y [$\lambda/D$]', fontsize=label_fs, labelpad=-5)
    # ax.set_xticks([])
    # ax.set_yticks([])

    ax = fig.add_subplot(gs[1, 0])
    # ax.imshow(np.angle(dJ_dE_DM), cmap='twilight',)
    ax.imshow(np.angle(dJ_dE_delA) * control_mask, cmap='twilight', extent=fp_extent)
    ax.set_title('Phase of Gradient\nat Focal Plane\n'+r'$\angle \frac{\partial J}{\partial \delta E} $', fontsize=title_fs)
    ax.set_xlabel('X [$\lambda/D$]', fontsize=label_fs)
    ax.set_ylabel('Y [$\lambda/D$]', fontsize=label_fs, labelpad=-5)
    # ax.set_xticks([])
    # ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(np.abs(dJ_dE_LP), cmap='plasma', extent=lyot_extent)
    ax.set_title('Amplitude of Gradient\nat Lyot Pupil\n'+r'$| \frac{\partial J}{\partial E_{LP}} |$', fontsize=title_fs)
    # ax.set_xlabel('X [mm]', fontsize=label_fs)
    ax.set_ylabel('Y [mm]', fontsize=label_fs, labelpad=0)
    # ax.set_xticks([])
    # ax.set_yticks([])

    ax = fig.add_subplot(gs[1, 1])
    ax.imshow(np.angle(dJ_dE_LP), cmap='twilight', extent=lyot_extent)
    ax.set_title('Phase of Gradient\nat Lyot Pupil\n'+r'$\angle \frac{\partial J}{\partial E_{LP}} $', fontsize=title_fs)
    ax.set_xlabel('X [mm]', fontsize=label_fs)
    ax.set_ylabel('Y [mm]', fontsize=label_fs, labelpad=0)
    # ax.set_xticks([])
    # ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(np.abs(dJ_dE_PUP), cmap='plasma', extent=pup_extent)
    ax.set_title('Amplitude of Gradient\nat Pre-FPM Pupil\n'+r'$| \frac{\partial J}{\partial E_{PUP}} |$', fontsize=title_fs)
    # ax.set_xlabel('X [mm]', fontsize=label_fs)
    ax.set_ylabel('Y [mm]', fontsize=label_fs, labelpad=0)
    # ax.set_xticks([])
    # ax.set_yticks([])

    ax = fig.add_subplot(gs[1, 2])
    ax.imshow(np.angle(dJ_dE_PUP), cmap='twilight', extent=pup_extent)
    ax.set_title('Phase of Gradient\nat Pre-FPM Pupil\n'r'$\angle \frac{\partial J}{\partial E_{PUP}} $', fontsize=title_fs)
    ax.set_xlabel('X [mm]', fontsize=label_fs)
    ax.set_ylabel('Y [mm]', fontsize=label_fs, labelpad=0)
    # ax.set_xticks([])
    # ax.set_yticks([])

    ax = fig.add_subplot(gs[:, 3])
    ax.imshow(dJ_dS_DM.real, cmap='viridis', extent=pup_extent)
    ax.set_title('Gradient at\nDM Surface\n'r'$ \frac{\partial J}{\partial S_{DM}} $', fontsize=title_fs)
    ax.set_xlabel('X [mm]', fontsize=label_fs)
    ax.set_ylabel('Y [mm]', fontsize=label_fs, labelpad=0)
    # ax.set_xticks([])
    # ax.set_yticks([])

    ax = fig.add_subplot(gs[:, 4])
    ax.imshow(dm_grad, cmap='viridis',)
    ax.set_title('Gradient at\nDM Actuators\n'r'$ \frac{\partial J}{\partial A} $', fontsize=title_fs)
    ax.set_xlabel('X [Actuators]', fontsize=label_fs)
    ax.set_ylabel('Y [Actuators]', fontsize=label_fs, labelpad=7.5)
    # ax.set_xticks([])
    # ax.set_yticks([])

    plt.subplots_adjust(
        wspace=wspace,
        hspace=hspace,
    )
    if fname is not None: fig.savefig(fname, format='pdf', bbox_inches="tight")




