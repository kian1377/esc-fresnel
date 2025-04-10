import numpy as np
import astropy.units as u

import poppy
from poppy.poppy_core import PlaneType
pupil = PlaneType.pupil
inter = PlaneType.intermediate
image = PlaneType.image

'''
Notes about the abbreviations used here: 
    ifp = intermediate focal plane
    fm = fold mirror
    km = K-mirror
    Roap = Relay OAP
    AOoap = AO bench OAP
    fpsm = focal plane steering mirror
    pfm = periscope fold mirror
    pupilSM = pupil steering mirror
''' 

'''
These are the corrections (or fudge-factors) calculated using the Fresnel model
in order to fix the issue with focal-planes not being a tight focus or pupil planes
not being reimaged correctly. 
'''

def oap_fl(ROC, OAD):
    VFL = ROC/2
    DEL = OAD**2 / (2*ROC)
    A   = VFL - DEL
    EFL = np.sqrt(A**2 + OAD**2)
    return EFL

def conic_fl(ROC, K, OAD):
    f_parent = ROC/2
    SAG = OAD**2 / ( ROC + np.sqrt(ROC**2 - (K+1)*OAD**2) )
    f_apparent = ( f_parent - SAG ) / np.cos(np.arctan(OAD/(f_parent - SAG)))
    return f_apparent

zemax_inds = {
    'm1': 4,
    'm2': 5,
    'm3': 6,
    'm4': 9,
    'wcc_fp': 12,
    'oap1': 21,
    'fm1': 23,
    'pupil_mask': 27,  
    'oap2': 30,
    'ifp1': 32, 
    'oap3': 35,
    'fsm': 38, 
    'fm2': 41,
    'oap4': 44, 
    'ifp2': 46,
    'oap5': 48,
    'dm': 50,
    'input_lp': 52, 
    'input_qwp': 54,
    'oap6': 60,
    'fpm': 64,
    'oap7': 70,
    'lyot_plane': 82,
    # 'oap8_locam': 81, 
    'oap9': 86,
    'fieldstop': 88,
    'collimator': 94,
    'filter': 96,
    'output_qwp': 102, 
    'output_lp': 104,
    'oap10': 108,
    'camsci': 114,
}

diams = {
    'pupil': 2.43 *u.m,
    'm1': (6.50 *u.m, 1.38*u.m), # M1 is a tuple because it has an inner and outer diameter
    'm2': 2 * 3.573626217651691E+002 *u.mm, 
    'm3': (2 * 350*u.mm, 2 * 270 *u.mm), # M3 is a tuple because it is a rectangle with different dimensions along x and y
    # 'm4': 2 * 4.816942571704584E+001 *u.mm, 
    'm4': 90 *u.mm, 
    'oap1':  50.8*u.mm, 
    'fm1': 50.8*u.mm, 
    'pupil_mask': 14*u.mm, 
    'oap2':  25.4*u.mm, 
    'oap3':  25.4*u.mm, 
    'fsm': 25.4*u.mm, 
    'fm2': 25.4*u.mm,
    'oap4':  25.4*u.mm, 
    'oap5':  25.4*u.mm, 
    'oap6':  25.4*u.mm, 
    'oap7':  25.4*u.mm, 
    'oap8_locam':  25.4*u.mm, 
    'oap9':  25.4*u.mm, 
    'collimator':25.4*u.mm,
    'oap10':  25.4*u.mm, 
    'input_lp': 25.4*u.mm, 
    'input_qwp': 25.4*u.mm, 
    'output_qwp': 25.4*u.mm, 
    'output_lp': 25.4*u.mm,
    'filter':25.4*u.mm, 
}

# using zemax distances
distances = {
    # 'm1-m2': 7.504933097983007E+003 *u.mm, 
    # 'm2-m3': 7.442924445155444E+003 *u.mm, 
    # 'm3-m4': 1.050017342363088E+003 *u.mm, 
    # 'm4-wcc_fp': 1.272898367248996E+003 *u.mm, 

    'm1-m2': 7.504933069210861E+003 *u.mm, 
    'm2-m3': 7.442924483744588E+003 *u.mm, 
    'm3-m4': 1.050017342144711E+003 *u.mm, 
    'm4-wcc_fp': 1.273532030002423E+003 *u.mm, 

    'wcc_fp-oap1': 5.105869985134304E+002 *u.mm, 
    'oap1-fm1': 1.911769991649780E+002 *u.mm, 
    'fm1-pupil_mask': 5.167086395386104E+002 *u.mm, 

    'pupil_mask-oap2': 7.873448943460244E+001 *u.mm, 
    'oap2-ifp1': 1.022237526492536E+002*u.mm, 
    'ifp1-oap3': 7.511509008726716E+001 *u.mm, 
    'oap3-fsm': 1.216640462422038E+002 *u.mm, 

    'fsm-fm2': 5.459237814349035E+001 *u.mm, 
    'fm2-oap4': 8.069749536074960E+001 *u.mm, 
    'oap4-ifp2': 1.129076777227092E+002 *u.mm, 
    'ifp2-oap5': 1.129076785559410E+002 *u.mm, 
    'oap5-dm': 6.714161347730624E+001 *u.mm, 
    'dm-input_lp': 1.234055034338526E+002 *u.mm, 
    'input_lp-input_qwp': 1.099859443087553E+001 *u.mm, 
    'input_qwp-oap6': 8.499825769977906E+001 *u.mm, 
    'oap6-fpm': 4.643321137882849E+002 *u.mm, 
    'fpm-oap7': 2.062190530703447E+002 *u.mm, 
    'oap7-lyot': 2.538047651128181E+002 *u.mm, 

    'lyot-oap9': 7.842937112339132E+001 *u.mm, 
    'oap9-fieldstop': 1.464285714571997E+002 *u.mm, 
    'fieldstop-collimator': 1.407867006808410E+002 *u.mm, 
    'collimator-filter': 3.192111065688332E+002 *u.mm, 
    'filter-output_qwp': 1.554068747912461E+002 *u.mm, 
    'output_qwp-output_lp': 1.061985976562573E+001 *u.mm, 
    'output_lp-oap10': 8.357002309866584E+001 *u.mm, 
    'oap10-scicam': 1.465794344290553E+002 *u.mm, 
}

parent_rocs = {
    'm1': 1.625600000000000E+004*u.mm, 
    'm2': 1.660574984313269E+003*u.mm, 
    'm3': 1.830517447403161E+003*u.mm, 
    'oap1': 1.000203185722913E+003*u.mm,  
    'oap2': 2.000000000000000E+002*u.mm,
    'oap3': 1.420000000000000E+002*u.mm,
    'oap4': 2.000000000000000E+002*u.mm, 
    'oap5': 2.000000000000000E+002*u.mm, 
    'oap6': 9.000000000000000E+002*u.mm, 
    'oap7': 4.000000000000000E+002*u.mm, 
    'oap8_locam': 2.000000000000000E+002*u.mm,
    'oap9': 2.800000000000000E+002*u.mm, 
    'collimator': 2.800000000000000E+002*u.mm, 
    'oap10': 2.800000000000000E+002 *u.mm, 
}

conics = {
    'm1': -9.953570000000001E-001,
    'm2': -1.566500000000000E+000,
    'm3': -7.180000000000000E-001,
    'oap1': -1, 
    'oap2': -1, 
    'oap3': -1, 
    'oap4': -1, 
    'oap5': -1, 
    'oap6': -1, 
    'oap7': -1, 
    'oap8_locam': -1, 
    'oap9': -1, 
    'collimator': 0, 
    'oap10': -1, 
}

off_axis_distances = {
    'm1': np.sqrt(9.746832E+02**2 + 1.666386E+03**2)*u.mm,
    'm2': np.sqrt(1.020295E+02**2 + 1.272150E+02**2)*u.mm,
    'm3': np.sqrt(1.325541E+02**2 + (3.991276E+01 + 240)**2)*u.mm,
    'oap1': np.sqrt( (2.229119E+00 + 68.4)**2 + (8.220046E+00 + 118)**2 )*u.mm,
    'oap2': np.sqrt((-30)**2 + (0)**2 )*u.mm,
    'oap3': np.sqrt( (34.2)**2 + (0)**2 )*u.mm,
    'oap4': np.sqrt(  (-72)**2  )*u.mm,
    'oap5': np.sqrt(  (72)**2  )*u.mm,
    'oap6': np.sqrt(  (151)**2  )*u.mm,
    'oap7': np.sqrt(  (70.5)**2  )*u.mm,
    'oap8_locam': np.sqrt(  72**2  )*u.mm,
    'oap9': np.sqrt(  (-60)**2  )*u.mm,
    'collimator': np.sqrt(  (21)**2  )*u.mm,
    'oap10': np.sqrt(  (-60)**2  )*u.mm,
}

apparent_fls_theo = {
    'm1': conic_fl(parent_rocs['m1'], conics['m1'], off_axis_distances['m1']), 
    'm2': -conic_fl(parent_rocs['m2'], conics['m2'], off_axis_distances['m2']),
    'm3': conic_fl(parent_rocs['m3'], conics['m3'], off_axis_distances['m3']),
    'oap1': conic_fl(parent_rocs['oap1'], conics['oap1'], off_axis_distances['oap1']),
    'oap2': conic_fl(parent_rocs['oap2'], conics['oap2'], off_axis_distances['oap2']),  
    'oap3': conic_fl(parent_rocs['oap3'], conics['oap3'], off_axis_distances['oap3']),
    'oap4': conic_fl(parent_rocs['oap4'], conics['oap4'], off_axis_distances['oap4']),
    'oap5': conic_fl(parent_rocs['oap5'], conics['oap5'], off_axis_distances['oap5']),
    'oap6':  conic_fl(parent_rocs['oap6'], conics['oap6'], off_axis_distances['oap6']),
    'oap7':  conic_fl(parent_rocs['oap7'], conics['oap7'], off_axis_distances['oap7']),
    'oap8_locam':  conic_fl(parent_rocs['oap8_locam'], conics['oap8_locam'], off_axis_distances['oap8_locam']),
    'oap9':  conic_fl(parent_rocs['oap9'], conics['oap9'], off_axis_distances['oap9']),
    'collimator':  conic_fl(parent_rocs['collimator'], conics['collimator'], off_axis_distances['collimator']),
    'oap10': conic_fl(parent_rocs['oap10'], conics['oap10'], off_axis_distances['oap10']),
}

apparent_fls = {
    # 'm1': conic_fl(parent_rocs['m1'], conics['m1'], off_axis_distances['m1']) + 1.75*u.mm, 
    # 'm2': -conic_fl(parent_rocs['m2'], conics['m2'], off_axis_distances['m2']) - 7.45*u.mm,
    # 'm3': conic_fl(parent_rocs['m3'], conics['m3'], off_axis_distances['m3']) - 8.05*u.mm,
    'm1': 8.242749125291561E+003*u.mm,
    'm2': -8.437228211011820E+002*u.mm,
    'm3': 9.351433621191690E+002*u.mm,
    'oap1': distances['wcc_fp-oap1'],
    'oap2': distances['oap2-ifp1'],  
    'oap3': distances['ifp1-oap3'], 
    'oap4': distances['oap4-ifp2'], 
    'oap5': distances['ifp2-oap5'], 
    'oap6': distances['oap6-fpm'], 
    'oap7': distances['fpm-oap7'], 
    'oap8_locam':  conic_fl(parent_rocs['oap8_locam'], conics['oap8_locam'], off_axis_distances['oap8_locam']),
    'oap9':  distances['oap9-fieldstop'],
    'collimator':  distances['fieldstop-collimator'],
    'oap10': distances['oap10-scicam'],
}

elements = {
    'm1': poppy.QuadraticLens(f_lens=apparent_fls['m1'], name='m1', planetype=pupil),
    'm2': poppy.QuadraticLens(f_lens=apparent_fls['m2'], name='m2', planetype=inter),
    'm3': poppy.QuadraticLens(f_lens=apparent_fls['m3'], name='m3', planetype=inter),
    # 'm4': poppy.CircularAperture(radius=diams['m4']/2, name='m4 (pupil)', planetype=pupil),
    'wcc_fp': poppy.ScalarTransmission(name='wcc_fp', planetype=inter), 
    'oap1': poppy.QuadraticLens(f_lens=apparent_fls['oap1'], name='oap1', planetype=inter),
    'fm1': poppy.ScalarTransmission(name='fm1', planetype=inter),
    'pupil_mask': poppy.CircularAperture(radius=diams['pupil_mask']/2, name='pupil_mask', planetype=pupil), 
    'oap2': poppy.QuadraticLens(f_lens=apparent_fls['oap2'], name='oap2', planetype=inter),
    'ifp1': poppy.ScalarTransmission(name='ifp1', planetype=inter),
    'oap3': poppy.QuadraticLens(f_lens=apparent_fls['oap3'], name='oap3', planetype=inter),
    # 'fsm': poppy.ScalarTransmission(name='FSM (pupilish)', planetype=pupil),
    # 'fsm': poppy.CircularAperture(radius=25.4/2 * u.mm, name='fsm (pupilish)', planetype=pupil),
    'fm2': poppy.ScalarTransmission(name='fm2', planetype=inter),
    'oap4': poppy.QuadraticLens(f_lens=apparent_fls['oap4'], name='oap4', planetype=inter),
    'ifp2': poppy.ScalarTransmission(name='ifp2', planetype=inter),
    'oap5': poppy.QuadraticLens(f_lens=apparent_fls['oap5'], name='oap5', planetype=inter),
    'input_lp': poppy.CircularAperture(radius=25.4*u.mm/2, name='input_lp', planetype=inter),
    'input_qwp': poppy.CircularAperture(radius=25.4*u.mm/2, name='input_qwp', planetype=inter),
    'oap6': poppy.QuadraticLens(f_lens=apparent_fls['oap6'], name='oap6', planetype=inter),
    'fpm': poppy.ScalarTransmission(name='fpm', planetype=inter),
    'oap7': poppy.QuadraticLens(f_lens=apparent_fls['oap7'], name='oap7', planetype=inter),
    'lyot_plane': poppy.ScalarTransmission(name='lyot_plane (pupil)', planetype=pupil), 
    'oap8_locam': poppy.QuadraticLens(f_lens=apparent_fls['oap8_locam'], name='oap8_locam', planetype=inter),
    'oap9': poppy.QuadraticLens(f_lens=apparent_fls['oap9'], name='oap9', planetype=inter),
    'fieldstop': poppy.ScalarTransmission(name='fieldstop (ifp3)', planetype=inter),
    'collimator': poppy.QuadraticLens(f_lens=apparent_fls['collimator'], name='collimator', planetype=inter),
    'filter': poppy.ScalarTransmission(name='filter', planetype=inter),
    'output_qwp': poppy.CircularAperture(radius=25.4*u.mm/2, name='output_qwp', planetype=inter),
    'output_lp': poppy.CircularAperture(radius=25.4*u.mm/2, name='output_lp', planetype=inter),
    'oap10': poppy.QuadraticLens(f_lens=apparent_fls['oap10'], name='oap10', planetype=inter),
    # 'camsci': poppy.ScalarTransmission(name='camsci (fp)', planetype=inter),
}

zemax_footprint_diams = {
    'pupil_mask_x': 2* 6.37*u.mm,
    'pupil_mask_y': 2* 6.35*u.mm,
    'M4_x': 2*15.921*u.mm,
    'M4_y': 2*16.399 * u.mm,
    'FSM_x': 2*4.765*u.mm,
    'FSM_y': 2*4.643*u.mm,
    'DM_x': 2*4.728*u.mm,
    'DM_y': 2*4.6425*u.mm,
    'lyot_plane_x': 2*2.095*u.mm,
    'lyot_plane_y': 2*2.072*u.mm,
}

zemax_airy_rads = {
    'wcc_fp':31.752*u.um,
    # 'ifp1': *u.um,
    # 'ifp2': *u.um,
    # 'fpm': *u.um,
    # 'fieldstop': *u.um,
    'scicam': 9.126*u.um,
}





