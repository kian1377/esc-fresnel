from .math_module import xp, xcipy, ensure_np_array
from . import utils

import numpy as np
import scipy
import astropy.units as u
from astropy.io import fits
from pathlib import Path

import poppy

class DeformableMirror(poppy.AnalyticOpticalElement):
    
    def __init__(
            self,
            inf_fun,
            inf_sampling,
            Nact=34,
            act_spacing=300e-6*u.m,
            shift=np.array([0, 0])*u.m,
            aperture=None,
            include_reflection=True,
            planetype=poppy.poppy_core.PlaneType.intermediate,
            name='DM',
        ):
        
        self.inf_fun = inf_fun
        self.inf_sampling = inf_sampling

        self.Nact = Nact
        self.act_spacing = act_spacing
        self.include_reflection = include_reflection
        self.shift = shift

        self.Nsurf = inf_fun.shape[0]
        self.pixelscale = self.act_spacing/(self.inf_sampling*u.pix)
        self.active_diam = self.Nact * self.act_spacing

        self.yc, self.xc = (xp.indices((Nact, Nact)) - Nact//2 + 1/2)
        self.rc = xp.sqrt(self.xc**2 + self.yc**2)
        self.dm_mask = self.rc<(Nact/2 + 1/2)
        self.Nacts = int(xp.sum(self.dm_mask))
        
        self.command = xp.zeros((self.Nact, self.Nact))
        self.actuators = xp.zeros(self.Nacts)

        self.aperture = aperture
        self.planetype = planetype
        self.name = name

        self.inf_fun_fft = xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift(self.inf_fun,)))
        fx = xp.fft.fftshift(xp.fft.fftfreq(self.Nsurf))
        fy = xp.fft.fftshift(xp.fft.fftfreq(self.Nsurf))
        x = self.inf_sampling*(xp.linspace(-self.Nact//2, self.Nact//2-1, self.Nact) + 1/2)
        y = self.inf_sampling*(xp.linspace(-self.Nact//2, self.Nact//2-1, self.Nact) + 1/2)

        self.Mx = xp.exp(-1j*2*np.pi*xp.outer(fx,x))
        self.My = xp.exp(-1j*2*np.pi*xp.outer(y,fy))
        self.Mx_back = xp.exp(1j*2*np.pi*xp.outer(x,fx)) # adjoint DM model MFT matrices
        self.My_back = xp.exp(1j*2*np.pi*xp.outer(fy,y))

        self.dm_channels = xp.zeros((10,Nact,Nact))
        self.total_command = xp.sum(self.dm_channels, axis=0)

        self.pxscl_tol = 1e-6

    def set_command(self, command, channel=1):
        command *= self.dm_mask
        self.dm_channels[channel] = command
        self.total_command = xp.sum(self.dm_channels, axis=0)

    def add_command(self, command, channel=1):
        command *= self.dm_mask
        self.dm_channels[channel] = self.dm_channels[channel] + command
        self.total_command = xp.sum(self.dm_channels, axis=0)

    def zero_all_channels(self,):
        self.dm_channels = xp.zeros((10,self.Nact,self.Nact))
        self.total_command = xp.sum(self.dm_channels, axis=0)

    def get_command(self, channel=1):
        return self.dm_channels[channel]

    def get_surface(self):
        mft_command = self.Mx @ self.total_command @ self.My
        fourier_surf = self.inf_fun_fft * mft_command
        surf = xp.fft.fftshift( xp.fft.ifft2( xp.fft.ifftshift( fourier_surf ))).real
        shift_pix = self.shift.to_value(u.m) / self.pixelscale.to_value(u.m/u.pix)
        surf = xcipy.ndimage.shift(surf, xp.flip(shift_pix), order=3)
        return surf
    
    # METHODS TO BE COMPATABLE WITH POPPY
    def get_opd(self, wave):
        opd = self.get_surface()
        if self.include_reflection:
            opd *= 2

        pxscl_diff = wave.pixelscale.to_value(u.m/u.pix) - self.pixelscale.to_value(u.m/u.pix) 
        if pxscl_diff < self.pxscl_tol:
            opd = utils.interp_arr(opd, self.pixelscale.to_value(u.m/u.pix), wave.pixelscale.to_value(u.m/u.pix) )
        
        opd = utils.pad_or_crop(opd, wave.shape[0])

        return opd

    def get_transmission(self, wave):
        if self.aperture is None:
            trans = xp.ones_like(wave.wavefront)
        else:
            trans = self.aperture.get_transmission(wave)
        return trans
    
    def get_phasor(self, wave):
        assert (wave.planetype != poppy.poppy_core.PlaneType.image)

        dm_phasor = self.get_transmission(wave) * xp.exp(1j * 2*np.pi/wave.wavelength.to_value(u.m) * self.get_opd(wave))

        return dm_phasor



