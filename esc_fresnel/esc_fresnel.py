from .math_module import xp, xcipy, ensure_np_array
import esc_fresnel.utils as utils
import esc_fresnel.props as props
import esc_fresnel.dm as dm
import esc_fresnel.imshows as imshows
from . import esc_optics

import numpy as np
import astropy.units as u
from astropy.io import fits
import time
import os
from pathlib import Path
import copy
from scipy.signal import windows

import poppy
from poppy.poppy_core import PlaneType
pupil = PlaneType.pupil
inter = PlaneType.intermediate
image = PlaneType.image

class single():

    def __init__(
            self, 
            wavelength=650e-9*u.m, 
            npix=1000, 
            oversample=2.048, 
            npsf=256,
            dm_ref=xp.zeros((34,34)),
            use_corrections=True,
        ):
        
        self.wavelength = wavelength
        
        self.pupil_diam = 2.43*u.m
        self.lyot_stop_diam = 3.7*u.mm
        self.wavelength_c = 650e-9*u.m
        self.camsci_pxscl = 3.76*u.um/u.pix

        # The following quantities are computed using the Fresnel model and hard-coded in
        self.dm_pupil_diam = 9.351 * u.mm
        self.lyot_pupil_diam = 4.153042573359455 * u.mm
        self.final_pupil_diam = 3.9930329934638595 * u.mm
        self.final_fl = esc_optics.apparent_fls['oap10']
        self.fpm_fl = esc_optics.apparent_fls['oap6']
        
        self.lyot_ratio = (self.lyot_stop_diam / self.lyot_pupil_diam).decompose().value
        self.lyot_to_final_mag = self.final_pupil_diam / self.lyot_pupil_diam
        self.exit_pupil_diam = self.lyot_stop_diam * self.lyot_to_final_mag
        self.camsci_pxscl_lamDc = (self.camsci_pxscl / (self.final_fl * self.wavelength_c / self.exit_pupil_diam)).decompose().value
        self.camsci_airy_rad = 1.22 * (self.wavelength_c * self.final_fl / self.exit_pupil_diam).to(u.mm)

        self.wcc_efl = 96551.6*u.mm
        self.wcc_fnum = 40.0402
        self.wcc_airy_diam = 31.752*u.um
        
        self.m4_corr = 0*u.mm
        self.pupil_mask_corr = 0*u.mm
        self.fsm_corr = 0*u.mm
        self.dm_corr = 0*u.mm
        self.lyot_corr = 0*u.mm

        if use_corrections:
            self.wcc_corr = -0.0013996804923976924 *u.mm
            self.ifp1_corr = -2.7895964649360394e-05 *u.mm
            self.ifp2_corr = -0.00012892930811858605 *u.mm
            self.fpm_corr = -0.00896733376620773 *u.mm
            self.fieldstop_corr = -0.0072283463943279*u.mm
            self.scicam_corr = -0.008480730812721049*u.mm
        else:
            self.wcc_corr = 0*u.mm
            self.ifp1_corr = 0*u.mm
            self.ifp2_corr = 0*u.mm
            self.fpm_corr = 0*u.mm
            self.fieldstop_corr = 0*u.mm
            self.scicam_corr = 0*u.mm

        self.use_vortex = False
        self.use_lyot = True
        self.plot_vortex = False

        self.wfes = {}

        self.Imax_ref = 1

        self.npsf = npsf
        self.npix = npix
        self.oversample = oversample
        self.N = int(self.npix*self.oversample)

        self.EP = poppy.CircularAperture(
            radius=self.pupil_diam/2, 
            name='Entrance Pupil (subaperture)', 
            planetype=pupil,
        )

        self.LYOT = poppy.CircularAperture(
            radius=self.lyot_stop_diam/2, 
            name='Lyot Stop (pupil)',
        )

        self.DOPD = poppy.ArrayOpticalElement(
            opd=xp.zeros((self.npix, self.npix)), 
            pixelscale=self.pupil_diam/(self.npix*u.pix), 
            planetype=inter, 
            name='Dummy Pupil OPD',
        )

        self.GAP_MASK = self.EP.get_transmission(poppy.FresnelWavefront(beam_radius=self.pupil_diam/2, npix=self.npix, oversample=1))
        self.BAP_MASK = self.GAP_MASK>0

        # VORTEX MODELING PARAMETERS
        self.oversample_vortex = 4.096
        self.N_vortex_lres = int(self.npix*self.oversample_vortex)
        self.vortex_win_diam = 30 # diameter of the window to apply with the vortex model
        self.lres_sampling = 1/self.oversample_vortex # low resolution sampling in lam/D per pixel
        self.lres_win_size = int(self.vortex_win_diam/self.lres_sampling)
        w1d = xp.array(windows.tukey(self.lres_win_size, 1, False))
        self.lres_window = utils.pad_or_crop(xp.outer(w1d, w1d), self.N_vortex_lres)
        self.vortex_lres = props.make_vortex_phase_mask(self.N_vortex_lres)
        # y,x = (xp.indices((self.N_vortex_lres, self.N_vortex_lres)) - self.N_vortex_lres//2)*self.lres_sampling
        # r = xp.sqrt(x**2 + y**2)
        # self.lres_dot_mask = r>=0.71/2

        self.hres_sampling = 0.025 # lam/D per pixel; this value is chosen empirically
        self.N_vortex_hres = int(np.round(self.vortex_win_diam/self.hres_sampling))
        self.hres_win_size = int(self.vortex_win_diam/self.hres_sampling)
        w1d = xp.array(windows.tukey(self.hres_win_size, 1, False))
        self.hres_window = utils.pad_or_crop(xp.outer(w1d, w1d), self.N_vortex_hres)
        self.vortex_hres = props.make_vortex_phase_mask(self.N_vortex_hres)

        y,x = (xp.indices((self.N_vortex_hres, self.N_vortex_hres)) - self.N_vortex_hres//2)*self.hres_sampling
        r = xp.sqrt(x**2 + y**2)
        self.hres_dot_mask = r>=0.3/2

        '''
        TODO:
            1. Add DM quantization errors
            2. Add pre-FPM low orders from Zemax
            3. Add post-FPM low orders from Zemax
        '''

        self.Nact = 34
        act_spacing = 300e-6*u.m
        self.dm_pxscl = self.dm_pupil_diam.to_value(u.m)/self.npix
        inf_sampling = act_spacing.to_value(u.m)/self.dm_pxscl
        inf_fun = utils.make_gaussian_inf_fun(
            act_spacing=act_spacing.to_value(u.m), 
            sampling=inf_sampling, 
            coupling=0.15, 
            Nact=self.Nact+2,
        )
        self.DM = dm.DeformableMirror(
            inf_fun=inf_fun, 
            inf_sampling=inf_sampling, 
            name='DM (pupil)',
        )

        self.dm_mask = self.DM.dm_mask
        self.Nacts = self.DM.Nacts
        self.dm_ref = dm_ref
        self.set_dm(dm_ref, channel=0)

        self.return_pupil = False

        self.source_offset = (0,0)
        self.as_per_lamD = ((self.wavelength_c/(self.pupil_diam*self.lyot_ratio))*u.radian).to(u.arcsec)

        self.det_rotation = 0

    # useful for parallelization with ray ACTORS
    def getattr(self, attr):
        return getattr(self, attr)
    
    def setattr(self, attr, val):
        setattr(self, attr, val)

    def zero_dm(self):
        self.DM.zero_all_channels()

    def reset_dm(self):
        self.DM.zero_all_channels()
        self.DM.set_command(self.dm_ref, channel=0)
        
    def set_dm(self, command, channel=1):
        self.DM.set_command(command, channel=channel)
        
    def add_dm(self, command, channel=1):
        self.DM.add_command(command, channel=channel)
        
    def get_dm(self, channel=1):
        return self.DM.get_command(channel=channel)
    
    def init_fosys(self):

        fosys1 = poppy.FresnelOpticalSystem(pupil_diameter=self.pupil_diam, npix=self.npix, beam_ratio=1/self.oversample, name='ESC to FPM')
        fosys1.add_optic(self.EP)
        fosys1.add_optic(self.DOPD)
        fosys1.add_optic(esc_optics.elements['m1'])
        if self.wfes.get('m1') is not None: fosys1.add_optic(self.wfes['m1'])
        fosys1.add_optic(esc_optics.elements['m2'], distance=esc_optics.distances['m1-m2'])
        if self.wfes.get('m2') is not None: fosys1.add_optic(self.wfes['m2'])
        fosys1.add_optic(esc_optics.elements['m3'], distance=esc_optics.distances['m2-m3'])
        if self.wfes.get('m3') is not None: fosys1.add_optic(self.wfes['m3'])
        fosys1.add_optic(esc_optics.elements['m4'], distance=esc_optics.distances['m3-m4'] + self.m4_corr)
        if self.wfes.get('m4') is not None: fosys1.add_optic(self.wfes['m4'])
        fosys1.add_optic(esc_optics.elements['wcc_fp'], distance=esc_optics.distances['m4-wcc_fp'] - self.m4_corr + self.wcc_corr)
        fosys1.add_optic(esc_optics.elements['oap1'], distance=esc_optics.distances['wcc_fp-oap1'] - self.wcc_corr)
        if self.wfes.get('oap1') is not None: fosys1.add_optic(self.wfes['oap1'])
        fosys1.add_optic(esc_optics.elements['fm1'], distance=esc_optics.distances['oap1-fm1'])
        if self.wfes.get('fm1') is not None: fosys1.add_optic(self.wfes['fm1'])
        fosys1.add_optic(esc_optics.elements['pupil_mask'], distance=esc_optics.distances['fm1-pupil_mask'] + self.pupil_mask_corr)
        fosys1.add_optic(esc_optics.elements['oap2'], distance=esc_optics.distances['pupil_mask-oap2'] - self.pupil_mask_corr)
        if self.wfes.get('oap2') is not None: fosys1.add_optic(self.wfes['oap2'])
        fosys1.add_optic(esc_optics.elements['ifp1'], distance=esc_optics.distances['oap2-ifp1'] + self.ifp1_corr)
        fosys1.add_optic(esc_optics.elements['oap3'], distance=esc_optics.distances['ifp1-oap3'] - self.ifp1_corr)
        if self.wfes.get('oap3') is not None: fosys1.add_optic(self.wfes['oap3'])
        fosys1.add_optic(esc_optics.elements['fsm'], distance=esc_optics.distances['oap3-fsm'] + self.fsm_corr)
        if self.wfes.get('fsm') is not None: fosys1.add_optic(self.wfes['fsm'])
        fosys1.add_optic(esc_optics.elements['fm2'], distance=esc_optics.distances['fsm-fm2'] - self.fsm_corr)
        if self.wfes.get('fm2') is not None: fosys1.add_optic(self.wfes['fm2'])
        fosys1.add_optic(esc_optics.elements['oap4'], distance=esc_optics.distances['fm2-oap4'])
        if self.wfes.get('oap4') is not None: fosys1.add_optic(self.wfes['oap4'])
        fosys1.add_optic(esc_optics.elements['ifp2'], distance=esc_optics.distances['oap4-ifp2'] + self.ifp2_corr)
        fosys1.add_optic(esc_optics.elements['oap5'], distance=esc_optics.distances['ifp2-oap5'] - self.ifp2_corr)
        if self.wfes.get('oap5') is not None: fosys1.add_optic(self.wfes['oap5'])
        fosys1.add_optic(self.DM, distance=esc_optics.distances['oap5-dm'] + self.dm_corr)
        if self.wfes.get('DM') is not None: fosys1.add_optic(self.wfes['DM'])
        fosys1.add_optic(esc_optics.elements['input_lp'], distance=esc_optics.distances['dm-input_lp'] - self.dm_corr)
        if self.wfes.get('input_lp') is not None: fosys1.add_optic(self.wfes['input_lp'])
        fosys1.add_optic(esc_optics.elements['input_qwp'], distance=esc_optics.distances['input_lp-input_qwp'])
        if self.wfes.get('input_qwp') is not None: fosys1.add_optic(self.wfes['input_qwp'])
        fosys1.add_optic(esc_optics.elements['oap6'], distance=esc_optics.distances['input_qwp-oap6'])
        if self.wfes.get('oap6') is not None: fosys1.add_optic(self.wfes['oap6'])
        fosys1.add_optic(esc_optics.elements['fpm'], distance=esc_optics.distances['oap6-fpm'] + self.fpm_corr)

        fosys2 = poppy.FresnelOpticalSystem(npix=self.npix, beam_ratio=1/self.oversample, name='ESC Post-FPM')
        fosys2.add_optic(poppy.ScalarTransmission(name='Post FPM WF'))
        fosys2.add_optic(esc_optics.elements['oap7'], distance=esc_optics.distances['fpm-oap7'] - self.fpm_corr)
        if self.wfes.get('oap7') is not None: fosys2.add_optic(self.wfes['oap7'])
        fosys2.add_optic(esc_optics.elements['lyot_plane'], distance=esc_optics.distances['oap7-lyot'] + self.lyot_corr)
        if self.return_pupil:
            return fosys1, fosys2
        if self.use_lyot: fosys2.add_optic(self.LYOT)
        fosys2.add_optic(esc_optics.elements['oap9'], distance=esc_optics.distances['lyot-oap9'] - self.lyot_corr)
        if self.wfes.get('oap9') is not None: fosys2.add_optic(self.wfes['oap9'])
        fosys2.add_optic(esc_optics.elements['fieldstop'], distance=esc_optics.distances['oap9-fieldstop'] + self.fieldstop_corr)
        fosys2.add_optic(esc_optics.elements['collimator'], distance=esc_optics.distances['fieldstop-collimator'] - self.fieldstop_corr)
        if self.wfes.get('collimator') is not None: fosys2.add_optic(self.wfes['collimator'])
        fosys2.add_optic(esc_optics.elements['filter'], distance=esc_optics.distances['collimator-filter'])
        if self.wfes.get('filter') is not None: fosys2.add_optic(self.wfes['filter'])
        fosys2.add_optic(esc_optics.elements['output_qwp'], distance=esc_optics.distances['filter-output_qwp'])
        if self.wfes.get('output_qwp') is not None: fosys2.add_optic(self.wfes['output_qwp'])
        fosys2.add_optic(esc_optics.elements['output_lp'], distance=esc_optics.distances['output_qwp-output_lp'])
        if self.wfes.get('output_lp') is not None: fosys2.add_optic(self.wfes['output_lp'])
        fosys2.add_optic(esc_optics.elements['oap10'], distance=esc_optics.distances['output_lp-oap10'])
        if self.wfes.get('oap10') is not None: fosys2.add_optic(self.wfes['oap10'])
        fosys2.add_optic(poppy.Rotation(self.det_rotation, units='degrees'))
        fosys2.add_optic(poppy.Detector(pixelscale=self.camsci_pxscl, fov_pixels=self.npsf, interp_order=3, name='   Camsci (FP)',), distance=esc_optics.distances['oap10-scicam'] + self.scicam_corr)

        return fosys1, fosys2
    
    def init_inwave(self):
        inwave = poppy.FresnelWavefront(
            beam_radius=self.pupil_diam/2, wavelength=self.wavelength,
            npix=self.npix, oversample=self.oversample,
        )
        
        if np.abs(self.source_offset[0])>0 or np.abs(self.source_offset[1])>0:
            inwave.tilt(Xangle=self.source_offset[0]*self.as_per_lamD, Yangle=self.source_offset[1]*self.as_per_lamD)

        return inwave
    
    def apply_vortex(self, fpwf):
        vpup_wf = props.fft(fpwf)

        if self.plot_vortex: 
            imshows.imshow2(
                xp.abs(vpup_wf), xp.angle(vpup_wf), 
                'Virtual Pupil Amplitude', 'Virtual Pupil Phase',
                npix=1*self.npix,
            )

        lres_wf = utils.pad_or_crop(vpup_wf, self.N_vortex_lres) # pad to the larger array for the low res propagation
        fp_wf_lres = props.fft(lres_wf)
        fp_wf_lres *= self.vortex_lres * (1 - self.lres_window) # apply low res (windowed) FPM
        pupil_wf_lres = props.ifft(fp_wf_lres)
        pupil_wf_lres = utils.pad_or_crop(pupil_wf_lres, self.N,)
        if self.plot_vortex: 
            imshows.imshow2(
                xp.abs(pupil_wf_lres), xp.angle(pupil_wf_lres), 
                'FFT Pupil Amplitude', 'FFT Pupil Phase', 
                npix=1*self.npix,
            )

        fp_wf_hres = props.mft_forward(vpup_wf, self.npix, self.N_vortex_hres, self.hres_sampling, convention='-', fp_centering='odd')
        fp_wf_hres *= self.vortex_hres * self.hres_window * self.hres_dot_mask # apply high res (windowed) FPM
        pupil_wf_hres = props.mft_reverse(fp_wf_hres, self.hres_sampling, self.npix, self.N, convention='+', fp_centering='odd')
        if self.plot_vortex: 
            imshows.imshow2(
                xp.abs(pupil_wf_hres), xp.angle(pupil_wf_hres), 
                'MFT Pupil Amplitude', 'MFT Pupil Phase',
                npix=1*self.npix,
            )

        post_vortex_vpup_wf = (pupil_wf_lres + pupil_wf_hres)
        if self.plot_vortex: 
            imshows.imshow2(
                xp.abs(post_vortex_vpup_wf), xp.angle(post_vortex_vpup_wf), 
                'Total Pupil Amplitude', 'Total Pupil Phase',
                npix=1*self.npix,
            )

        post_vortex_fpwf = props.ifft(post_vortex_vpup_wf)

        return post_vortex_fpwf

    def calc_wfs(self, quiet=False):
        self.return_pupil = False
        fosys_to_fpm, fosys_to_scicam = self.init_fosys()
        ep_inwave = self.init_inwave()
        _, wfs_to_fpm = fosys_to_fpm.calc_psf(inwave=ep_inwave, normalize='none', return_intermediates=True)
        fpm_inwave = copy.copy(wfs_to_fpm[-1])
        if self.use_vortex: 
            post_fpm_wf = self.apply_vortex(copy.copy(fpm_inwave.wavefront))
            fpm_inwave.wavefront = copy.copy(post_fpm_wf)
        _, wfs_to_scicam = fosys_to_scicam.calc_psf(inwave=fpm_inwave, normalize='none', return_intermediates=True)
        all_wfs = wfs_to_fpm + wfs_to_scicam
        return all_wfs
    
    def calc_wf(self): 
        self.return_pupil = False
        fosys_to_fpm, fosys_to_scicam = self.init_fosys()
        ep_inwave = self.init_inwave()
        _, fpm_wf = fosys_to_fpm.calc_psf(inwave=ep_inwave, normalize='none', return_final=True)
        fpm_inwave = copy.copy(fpm_wf[0])
        if self.use_vortex: 
            post_fpm_wf = self.apply_vortex(copy.copy(fpm_inwave.wavefront))
            fpm_inwave.wavefront = copy.copy(post_fpm_wf)
        _, final_wf = fosys_to_scicam.calc_psf(inwave=fpm_inwave, normalize='none', return_final=True)
        return final_wf[0].wavefront / xp.sqrt(self.Imax_ref)
    
    def snap(self):
        im = xp.abs(self.calc_wf())**2
        return im
    
    def calc_pupil(self):
        self.return_pupil = True
        fosys_to_fpm, fosys_to_pupil = self.init_fosys()

        ep_inwave = self.init_inwave()
        _, wfs_to_fpm = fosys_to_fpm.calc_psf(inwave=ep_inwave, normalize='none', return_final=True)
        fpm_inwave = copy.copy(wfs_to_fpm[-1])

        if self.use_vortex: 
            post_fpm_wf = self.apply_vortex(copy.copy(fpm_inwave.wavefront))
            fpm_inwave.wavefront = copy.copy(post_fpm_wf)

        _, pupil_wf = fosys_to_pupil.calc_psf(inwave=fpm_inwave, normalize='none', return_final=True)

        pupil = utils.pad_or_crop(pupil_wf[-1].wavefront, self.npix)
        amp = xp.abs(pupil) * self.GAP_MASK
        phs = xp.angle(pupil) * self.GAP_MASK

        return amp*xp.exp(1j*phs)

import ray

class parallel():
    '''
    This is a class that sets up the parallelization of calc_psf such that it 
    we can generate polychromatic wavefronts that are then fed into the 
    various wavefront simulations.
    '''
    def __init__(
            self, 
            ACTORS,
        ):

        self.ACTORS = ACTORS
        self.NACTORS = len(ACTORS)
        self.wavelength_c = self.getattr('wavelength_c')

        self.npix = ray.get(ACTORS[0].getattr.remote('npix'))
        self.oversample = ray.get(ACTORS[0].getattr.remote('oversample'))
        
        self.camsci_pxscl = ray.get(ACTORS[0].getattr.remote('camsci_pxscl'))
        self.camsci_pxscl_lamDc = ray.get(ACTORS[0].getattr.remote('camsci_pxscl_lamDc'))
        self.npsf = ray.get(ACTORS[0].getattr.remote('npsf'))

        self.dm_mask = ray.get(ACTORS[0].getattr.remote('dm_mask'))
        self.Nact = self.dm_mask.shape[0]
        self.dm_ref = ray.get(ACTORS[0].getattr.remote('dm_ref'))

        self.Imax_ref = 1

    def getattr(self, attr):
        return ray.get(self.ACTORS[0].getattr.remote(attr))
    
    def setattr(self, attr, value):
        for i in range(len(self.ACTORS)):
            self.ACTORS[i].setattr.remote(attr,value)
    
    def reset_dm(self):
        for i in range(len(self.ACTORS)):
            self.ACTORS[i].reset_dm.remote()

    def zero_dm(self, channel=1):
        for i in range(len(self.ACTORS)):
            self.ACTORS[i].zero_dm.remote(channel)

    def set_dm(self, command, channel=1):
        for i in range(len(self.ACTORS)):
            self.ACTORS[i].set_dm.remote(command, channel)

    def add_dm(self, command, channel=1):
        for i in range(len(self.ACTORS)):
            self.ACTORS[i].add_dm.remote(command, channel)

    def get_dm(self, channel=1):
        return ray.get(self.ACTORS[0].get_dm.remote(channel))

    # def get_dm_total(self):
    #     return self.getattr('dm_total')

    def snap(self):
        pending_ims = []
        for i in range(self.NACTORS):
            future_ims = self.ACTORS[i].snap.remote()
            pending_ims.append(future_ims)
        ims = ray.get(pending_ims)
        ims = xp.array(ims)
        im = xp.mean(ims, axis=0)

        return im/self.Imax_ref






