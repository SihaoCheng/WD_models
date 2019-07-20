"""
This python module provides the tools to transform between broad-band photometry
and many white dwarf (WD) physical parameters (mass, cooling age, Teff, etc), 
based on an interpolation of cooling tracks from various existing WD models. In
particular, this module makes it easy to reading WD parameters according to its
coordinate on the *Gaia* H--R diagram. 

This module is written for python 3. To use this module, the fold "models/"
containing various existing cooling tracks should be downloaded together with
this script. This module is designed mainly for the following purposes:

1.  converting the coordinates of *Gaia* (and other) H--R diagram into WD 
    parameters;
2.  plotting contours of WD parameters on the *Gaia* (and other) H--R diagram.

The tools provided with the module also make it easy to transform between any
desired WD parameters. For details please refer to the doctrine of the 
functions. There are also descriptions and examples on my Github:
    https://github.com/SihaoCheng/WD_models
For questions or suggestions, please do not hesitate to contact me: 
    s.cheng@jhu.edu

18 Jul 2019
"""


import matplotlib.pyplot as plt
import numpy as np

from astropy.table import Table
from scipy.interpolate import CloughTocher2DInterpolator, LinearNDInterpolator
from scipy.interpolate import griddata, interp1d


#-------------------------------------------------------------------------------
#
#   Define the functions that will be used for reading cooling tracks and 
#   interpolating the mappings 
#
#-------------------------------------------------------------------------------


def interpolate_2d(x, y, z, method):
    if method == 'linear':
        interpolator = LinearNDInterpolator
    elif method == 'cubic':
        interpolator = CloughTocher2DInterpolator
    return interpolator((x,y), z, rescale=True)
    #return interp2d(x, y, z, kind=method)
  

def interp_atm(spec_type, color, logteff_logg_grid=(3.5, 5.1, 0.01, 6.5, 9.6, 0.01), 
               interp_type_atm='linear'):
    """interpolate the mapping (logteff, logg) --> photometry
    
    This function interpolates the mapping (logteff, logg) --> color index or
    bolometric correction (BC) of a passband.
    
    Args:
        spec_type:          String. {'H', 'He'}
            See the "Synthetic color" section on http://www.astro.umontreal.ca/~bergeron/CoolingModels/
        color:              String. {'G-Mbol', 'bp-rp', 'V-Mbol', 'u-g', 'B-z', etc.}
            The target color of the mapping. Any two bands between Gaia G, bp, rp,
            SDSS ugriz, and Johnson UBVRI, 2MASS JHK, and bolometric magnitude Mbol.
            For bolometric correction (BC), 'Mbol' must be after the minus sign '-'.
        logteff_logg_grid: (xmin, xmax, dx, ymin, ymax, dy). *Optional*
            corresponding to the grid of logTeff and logg.
        interp_type_atm:    String. {'linear', 'cubic'}. *Optional*
            Linear is much better for our purpose.
    
    Returns:
        grid_atm:           2d-array. 
                            The value of photometry on a (logteff, logg) grid
        atm_func:           Function. 
                            The interpolated mapping function:
                                (logteff, logg) --> photometry
        
    """
    logteff     = np.zeros(0) # atmospheric parameter
    logg        = np.zeros(0) # atmospheric parameter
    Mbol        = np.zeros(0) # Bolometric
    bp_Mag      = np.zeros(0) # Gaia
    rp_Mag      = np.zeros(0) # Gaia 
    G_Mag       = np.zeros(0) # Gaia
    u_Mag       = np.zeros(0) # SDSS
    g_Mag       = np.zeros(0) # SDSS
    r_Mag       = np.zeros(0) # SDSS
    i_Mag       = np.zeros(0) # SDSS
    z_Mag       = np.zeros(0) # SDSS
    U_Mag       = np.zeros(0) # Johnson
    B_Mag       = np.zeros(0) # Johnson
    V_Mag       = np.zeros(0) # Johnson
    R_Mag       = np.zeros(0) # Johnson
    I_Mag       = np.zeros(0) # Johnson
    J_Mag       = np.zeros(0) # 2MASS
    H_Mag       = np.zeros(0) # 2MASS
    K_Mag       = np.zeros(0) # 2MASS
        
    # read the table for all logg
    if spec_type == 'H':
        suffix = 'DA_thick'
    if spec_type == 'He':
        suffix = 'DB'
    Atm_color = Table.read('models/Montreal_atm_grid/Table_'+suffix,
                           format='ascii')
    selected  = Atm_color['Teff'] > 10**logteff_logg_grid[0]
    Atm_color = Atm_color[selected]
    
    # read columns
    logteff = np.concatenate(( logteff, np.log10(Atm_color['Teff']) ))
    logg    = np.concatenate(( logg, Atm_color['logg'] ))
    Mbol    = np.concatenate(( Mbol, Atm_color['Mbol'] ))
    bp_Mag  = np.concatenate(( bp_Mag, Atm_color['G_BP/R'] ))
    rp_Mag  = np.concatenate(( rp_Mag, Atm_color['G_RP/R'] ))
    G_Mag   = np.concatenate(( G_Mag, Atm_color['G/R'] ))
    u_Mag   = np.concatenate(( u_Mag, Atm_color['u'] ))
    g_Mag   = np.concatenate(( g_Mag, Atm_color['g'] ))
    r_Mag   = np.concatenate(( r_Mag, Atm_color['r'] ))
    i_Mag   = np.concatenate(( i_Mag, Atm_color['i'] ))
    z_Mag   = np.concatenate(( z_Mag, Atm_color['z'] ))
    U_Mag   = np.concatenate(( U_Mag, Atm_color['U'] ))
    B_Mag   = np.concatenate(( B_Mag, Atm_color['B'] ))
    V_Mag   = np.concatenate(( V_Mag, Atm_color['V'] ))
    R_Mag   = np.concatenate(( R_Mag, Atm_color['R'] ))
    I_Mag   = np.concatenate(( I_Mag, Atm_color['I'] ))
    J_Mag   = np.concatenate(( J_Mag, Atm_color['J'] ))
    H_Mag   = np.concatenate(( H_Mag, Atm_color['H'] ))
    K_Mag   = np.concatenate(( K_Mag, Atm_color['K'] ))
    
    # read the table for each mass
    # I suppose the color information in this table is from the interpolation
    # of the above table, so I do not need to read it.
    #for mass in ['0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0','1.2']:
    #    Atm_color = Table.read('models/Montreal_atm_grid/Table_Mass_' + mass +
    #                           '_'+spec_type, format='ascii')
    #    selected  = Atm_color['Teff'] > 10**logteff_logg_grid[0]
    #    Atm_color = Atm_color[selected]
    #    
    #    # read columns
    #    logteff = np.concatenate(( logteff, np.log10(Atm_color['Teff']) ))
    #    logg    = np.concatenate(( logg, Atm_color['logg'] ))
    #    Mbol    = np.concatenate(( Mbol, Atm_color['Mbol'] ))
    #    bp_Mag  = np.concatenate(( bp_Mag, Atm_color['G_BP/R'] ))
    #    rp_Mag  = np.concatenate(( rp_Mag, Atm_color['G_RP/R'] ))
    #    G_Mag   = np.concatenate(( G_Mag, Atm_color['G/R'] ))
    #    u_Mag   = np.concatenate(( u_Mag, Atm_color['u'] ))
    #    g_Mag   = np.concatenate(( g_Mag, Atm_color['g'] ))
    #    r_Mag   = np.concatenate(( r_Mag, Atm_color['r'] ))
    #    i_Mag   = np.concatenate(( i_Mag, Atm_color['i'] ))
    #    z_Mag   = np.concatenate(( z_Mag, Atm_color['z'] ))
    #    U_Mag   = np.concatenate(( U_Mag, Atm_color['U'] ))
    #    B_Mag   = np.concatenate(( B_Mag, Atm_color['B'] ))
    #    V_Mag   = np.concatenate(( V_Mag, Atm_color['V'] ))
    #    R_Mag   = np.concatenate(( R_Mag, Atm_color['R'] ))
    #    I_Mag   = np.concatenate(( I_Mag, Atm_color['I'] ))
    #    J_Mag   = np.concatenate(( J_Mag, Atm_color['J'] ))
    #    H_Mag   = np.concatenate(( H_Mag, Atm_color['H'] ))
    #    K_Mag   = np.concatenate(( K_Mag, Atm_color['K'] ))
    
    grid_x, grid_y = np.mgrid[logteff_logg_grid[0]:logteff_logg_grid[1]:logteff_logg_grid[2],
                              logteff_logg_grid[3]:logteff_logg_grid[4]:logteff_logg_grid[5]]
    
    # define the interpolation of mapping
    def interp(x, y, z, interp_type_atm='linear'):
        grid_z      = griddata(np.array((x, y)).T, z, (grid_x, grid_y),
                               method=interp_type_atm)
        z_func      = interpolate_2d(x, y, z, interp_type_atm)
        return grid_z, z_func
    
    division = color.find('-')
    if '-Mbol' in color:
        z = eval(color[:division] + '_Mag - Mbol')
    else:
        z = eval(color[:division] + '_Mag - ' + color[division+1:] + '_Mag')
    
    return interp(logteff, logg, z, interp_type_atm)


def read_cooling_tracks(low_mass_model, normal_mass_model, high_mass_model,
                        spec_type, logg_func=None, for_comparison=False):
    """ Read a set of cooling tracks
    
    This function reads the cooling models and stack together the data points
    of mass, logg, age, age_cool, logteff, and Mbol from different cooling
    tracks.
    
    Args:
        low_mass_model:     String. 
            Specifying the cooling model used for low-mass WDs (<~0.5Msun). 
            Its value should be one of the following: 
            ''                              no low-mass model will be read
            'Fontaine2001' or 'f'           the think-hydrogen CO WD model in 
                                            http://www.astro.umontreal.ca/~bergeron/CoolingModels/
            'Fontaine2001_thin' or 'ft'     the thin-hydrogen CO WD model in 
                                            http://www.astro.umontreal.ca/~bergeron/CoolingModels/
        normal_mass_model:  String. 
            Specifying the cooling model used for normal-mass WDs 
            (about 0.5~1.0Msun). Its value should be one of the following:
            ''                              no normal-mass model will be read
            'Fontaine2001' or 'f'           the think-hydrogen CO WD model in 
                                            http://www.astro.umontreal.ca/~bergeron/CoolingModels/
            'Fontaine2001_thin' or 'ft'     the thin-hydrogen CO WD model in 
                                            http://www.astro.umontreal.ca/~bergeron/CoolingModels/
            'Althaus2010_001' or 'a001'     Z=0.01, only for DA, http://evolgroup.fcaglp.unlp.edu.ar/TRACKS/tracks_cocore.html
            'Althaus2010_0001' or 'a0001'   Z=0.001, only for DA, http://evolgroup.fcaglp.unlp.edu.ar/TRACKS/tracks_cocore.html
            'Camisassa2017' or 'c'          only for DB, http://evolgroup.fcaglp.unlp.edu.ar/TRACKS/tracks_DODB.html
            'BaSTI' or 'b'                  with phase separation, Salaris et al. 2010, http://basti.oa-teramo.inaf.it
            'BaSTI_nosep' or 'bn'           no phase separation, Salaris et al. 2010, http://basti.oa-teramo.inaf.it
            'PG'                            only for DB
        high_mass_model:    String. 
            Specifying the cooling model used for high-mass WDs (>~1.0Msun). 
            Its value should be one of the following: 
            ''                              no high-mass model will be read
            'Fontaine2001' or 'f'           the think-hydrogen CO WD model in 
                                            http://www.astro.umontreal.ca/~bergeron/CoolingModels/
            'Fontaine2001_thin' or 'ft'     the thin-hydrogen CO WD model in 
                                            http://www.astro.umontreal.ca/~bergeron/CoolingModels/
            'ONe' or 'o'                    Camisassa et al. 2019, http://evolgroup.fcaglp.unlp.edu.ar/TRACKS/ultramassive.html
            'MESA' or 'm'                   Lauffer et al. 2019
            'BaSTI' or 'b'                  with phase separation, Salaris et al. 2010, http://basti.oa-teramo.inaf.it
            'BaSTI_nosep' or 'bn'           no phase separation, Salaris et al. 2010, http://basti.oa-teramo.inaf.it
        spec_type:          String. {'H', 'He'}
            Specifying the atmosphere composition.
            'H'                 pure-hydrogen atmosphere
            'He'                pure-helium atmosphere
        logg_func:          Function. *Optional*
            This is a function for (logteff, mass) --> logg. It is necessary 
            only for BaSTI models, because the BaSTI models do not directly 
            provide log g information.
        for_comparison:     Bool. *Optional*
            If true, more cooling tracks from different models will be used. 
            E.g., the Fontaine2001 model has m_WD = [..., 0.95, 1.00, ...], and
            the MESA model has m_WD = [1.0124, 1.019, ...]. If true, the 
            Fontaine2001 1.00Msun cooling track will be used; if false, it will
            not be used because it is too close to the MESA 1.0124Msun track.
    
    Returns:
        stacked data points from a set of cooling tracks.
        mass_array: 1d-array. The mass of WD in unit of solar mass. I only read
                    one value for a cooling track, not tracking the mass change.
        logg:       1d-array. in cm/s^2
        age:        1d-array. The total age of the WD in Gyr. Some are read
                    directly from the cooling tracks, but others are calculated
                    by assuming an initial--final mass relation (IFMR) of the WD
                    and adding the rough main-sequence age to the cooling age.
        age_cool:   1d-array. The cooling age of the WD in Gyr.
        logteff:    1d-array. The logarithm effective temperature of the WD in
                    Kelvin (K).
        Mbol:       1d-array. The absolute bolometric magnitude of the WD. Many
                    are converted from the log(L/Lsun) or log(L), where I adopt:
                            Mbol_sun = 4.75,
                            Lsun = 3.828e33 erg/s.
    
    """
    # determine which cooling tracks in a model to read
    mass_separation_1 = 0.45
    mass_separation_2 = 0.99
    if ('Althaus2010_' in normal_mass_model or 
        normal_mass_model == 'Camisassa2017' or 
        normal_mass_model == 'PG'
       ):
        if for_comparison == True:
            mass_seperation_1 = 0.501
        else:
            mass_seperation_1 = 0.45
    if 'BaSTI' in normal_mass_model:
        mass_seperation_1 = 0.501
    
    if high_mass_model == 'Fontaine2001' or high_mass_model == 'Fontaine2001_thin':
        mass_seperation_2 = 0.99
    if high_mass_model == 'ONe':
        mass_separation_2 = 1.09
    if high_mass_model == 'MESA':
        if for_comparison == True:
            mass_separation_2 = 1.01
        else:
            mass_separation_2 = 0.99
    if high_mass_model == 'BaSTI' or high_mass_model == 'BaSTInosep':
        mass_separation_2 = 0.99
    
    # define the initial-final mass relation for calculating the total age for 
    # some models
    IFMR        = interp1d((0.19, 0.3, 0.4, 0.50, 0.55, 0.65, 0.75, 0.85, 1.0,
                            1.25, 1.35),
                           (0.3, 0.5, 0.7, 0.95, 1, 2, 3, 3.5, 5, 8, 9),
                           fill_value = 0, bounds_error=False)
    t_index     = -3
    
    # initialize data points of cooling tracks
    logg        = np.zeros(0)
    age         = np.zeros(0) 
    age_cool    = np.zeros(0)
    logteff     = np.zeros(0)
    mass_array  = np.zeros(0)
    Mbol        = np.zeros(0)
    
    if spec_type == 'H':
        spec_suffix2 = 'DA'
    if spec_type == 'He':
        spec_suffix2 = 'DB'
    
    # read data from cooling models
    # Fontaine et al. 2001
    for mass in ['020','030','040','050','060','070','080','090','095','100',
                 '105','110','115','120','125','130']:
        if int(mass)/100 < mass_separation_1 and spec_type == 'H':
            if low_mass_model == 'Fontaine2001':
                spec_suffix = '0204'
            elif low_mass_model == 'Fontaine2001_thin':
                spec_suffix = '0210'
            else: continue
        elif (int(mass)/100 > mass_separation_1 and 
              int(mass)/100 < mass_separation_2 and spec_type == 'H'):
            if normal_mass_model == 'Fontaine2001':
                spec_suffix = '0204'
            elif normal_mass_model == 'Fontaine2001_thin':
                spec_suffix = '0210'
            else: continue
        elif int(mass)/100 > mass_separation_2 and spec_type == 'H':
            if high_mass_model == 'Fontaine2001':
                spec_suffix = '0204'
            elif high_mass_model == 'Fontaine2001_thin':
                spec_suffix = '0210'
            else: continue
        else: continue
        f       = open('models/Fontaine_AllSequences/CO_' + mass + 
                       spec_suffix)
        text    = f.read()
        example = ('      1    57674.0025    8.36722799  7.160654E+08 '
                    ' 4.000000E+05  4.042436E+33\n'
                    '        7.959696E+00  2.425570E+01  7.231926E+00 '
                    ' 0.0000000000  0.000000E+00\n'
                    '        6.019629E+34 -4.010597E+00 -1.991404E+00 '
                    '-3.055254E-01 -3.055254E-01'
                  )
        logg_temp       = []
        age_temp        = []
        age_cool_temp   = []
        logteff_temp    = []
        Mbol_temp       = []
        l_line          = len(example)
        for line in range(len(text)//l_line):
            logteff_temp.append(  np.log10(float(text[line*l_line+9:
                                                      line*l_line+21])) )
            logg_temp.append(     float(text[line*l_line+22:line*l_line+35]) )
            age_temp.append(      float(text[line*l_line+48:line*l_line+63]) +
                                  (IFMR(int(mass)/100))**(t_index) * 1e10 )
            age_cool_temp.append( float(text[line*l_line+48:line*l_line+63]) )
            Mbol_temp.append(     4.75 - 
                                  2.5 * np.log10(float(text[line*l_line+64:
                                                            line*l_line+76]
                                                      ) / 3.828e33) )
        mass_array  = np.concatenate(( mass_array, np.ones(len(age_temp)) * 
                                                   int(mass)/100 ))
        logg        = np.concatenate(( logg, logg_temp ))
        age         = np.concatenate(( age, age_temp ))
        age_cool    = np.concatenate(( age_cool, age_cool_temp ))
        logteff     = np.concatenate(( logteff, logteff_temp ))
        Mbol        = np.concatenate(( Mbol, Mbol_temp ))
        f.close()
    
    # define a smoothing function for future extension. Now it just returns the
    # input x vector.
    def smooth(x,window_len=5,window='hanning'):
        w=eval('np.'+window+'(window_len)')
        y=np.convolve(w/w.sum(),x,mode='same')
        return x
    
    # CO, DA (Althaus et al. 2010)
    if 'Althaus2010_' in normal_mass_model and spec_type == 'H':
        if '_001' in normal_mass_model:
            Althaus_masslist = ['0524','0570','0593','0609','0632','0659',
                                '0705','0767','0837','0877','0934']
            metallicity = '001'
        if '_0001' in normal_mass_model:
            Althaus_masslist = ['0505','0553','0593','0627','0660','0692',
                                '0863']
            metallicity = '0001'
        for mass in Althaus_masslist:
            Cool = Table.read('models/Althaus_2010_DA_CO/wdtracks_z' + 
                              metallicity + '/wd' + mass + '_z' + metallicity +
                              '.trk', format='ascii') 
            Cool = Cool[::1] #[(Cool['log(TEFF)'] > logteff_min) * (Cool['log(TEFF)'] < logteff_max)]
            mass_array  = np.concatenate(( mass_array, np.ones(len(Cool))*int(mass)/1000 ))
            logg        = np.concatenate(( logg, Cool['Log(grav)'] ))
            age         = np.concatenate(( age, Cool['age/Myr'] * 1e6 +
                                                (IFMR(int(mass)/1000))**(t_index)*1e10 ))
            age_cool    = np.concatenate(( age_cool, Cool['age/Myr'] * 1e6 ))
            logteff     = np.concatenate(( logteff, Cool['log(TEFF)'] ))
            Mbol        = np.concatenate(( Mbol, 4.75 - 2.5 * Cool['log(L)'] ))
            # additional
            logT_c      = Cool['logT_c']
            logrho_c    = Cool['logRo_c']
            mass_accurate= Cool['Mass']
            logL_nu     = Cool['Log(Lnu)']
            logMH       = Cool['LogMHtot']
            logr        = np.log10(Cool['R/R_sun'])
            del Cool
    
    # CO, DB (Camisassa et al. 2017)
    if normal_mass_model == 'Camisassa2017' and spec_type == 'He':
        for mass in ['051','054','058','066','074','087','100']:
            if int(mass)/100 < mass_separation_2:
                Cool = Table.read('models/Camisassa_2017_DB_CO/Z002/' + mass +
                                  'DB.trk', format='ascii') 
                dn = 1
                if int(mass)/100 > 0.95:
                    dn = 50
                Cool = Cool[::dn] # [(Cool['LOG(TEFF)'] > logteff_min) * (Cool['LOG(TEFF)'] < logteff_max)]
                #Cool.sort('Log(edad/Myr)')
                mass_array  = np.concatenate(( mass_array, np.ones(len(Cool)) * int(mass)/100 ))
                logg        = np.concatenate(( logg, Cool['Log(grav)'] ))
                age         = np.concatenate(( age, (10**Cool['Log(edad/Myr)'] -
                                                     10**Cool['Log(edad/Myr)'][0]) * 1e6 +
                                                    (IFMR(int(mass)/100))**(t_index) * 1e10 ))
                age_cool    = np.concatenate(( age_cool, (10**Cool['Log(edad/Myr)'] -
                                                          10**Cool['Log(edad/Myr)'][0]) * 1e6 ))
                logteff     = np.concatenate(( logteff, Cool['LOG(TEFF)'] ))
                Mbol        = np.concatenate(( Mbol, 4.75 - 2.5 * Cool['LOG(L)'] ))
                # additional
                logT_c      = Cool['T_c'] + 6
                logrho_c    = Cool['Ro_c']
                XH_c        = Cool['Hc']
                XHe_c       = Cool['Hec']
                mass_accurate= Cool['Masa']
                logL_nu     = Cool['Log(Lnu)']
                logMH       = Cool['LogMHtot']
                logHeBuf    = Cool['LogHeBuf']
                logr        = np.log10(Cool['R/R_sun'])
                L_LH        = Cool['L.H.[erg/s)]'] / 3.828e33
                L_PS        = Cool['Sep.Fase[erg/s]'] / 3.828e33
                del Cool
    
    # BaSTI model
    for mass in ['054','055','061','068','077','087','100','110','120']:
        if int(mass)/100 < mass_separation_2:
            if normal_mass_model == 'BaSTI':
                sep = 'sep'
            elif normal_mass_model == 'BaSTI_nosep':
                sep = 'nosep'
            else: continue
        elif int(mass)/100 > mass_separation_2:
            if high_mass_model == 'BaSTI':
                sep = 'sep'
            elif high_mass_model == 'BaSTI_nosep':
                sep = 'nosep'
            else: continue
        else: continue
        Cool = Table.read('models/BaSTI/COOL' + mass + 'BaSTIfinale' + 
               spec_suffix2 + sep +'.sdss', format='ascii')
        dn = 1
        if int(mass)/100 > 1.05:
            dn = 5
        Cool = Cool[::dn] # [(Cool['log(Teff)'] > logteff_min) * (Cool['log(Teff)'] < logteff_max)]
        #Cool.sort('Log(edad/Myr)')
        Cool['Log(grav)'] = logg_func(Cool['log(Teff)'], np.ones(len(Cool)) * int(mass)/100)
        mass_array  = np.concatenate(( mass_array, np.ones(len(Cool)) * int(mass)/100 ))
        logg        = np.concatenate(( logg, Cool['Log(grav)'] ))
        age         = np.concatenate(( age, 10**Cool['log(t)'] - 10**Cool['log(t)'][0] +
                                            (IFMR(int(mass)/100))**(t_index) * 1e10 ))
        age_cool    = np.concatenate(( age_cool, 10**Cool['log(t)'] - 10**Cool['log(t)'][0]  ))
        logteff     = np.concatenate(( logteff, Cool['log(Teff)'] ))
        Mbol        = np.concatenate(( Mbol, 4.75 - 2.5 * Cool['log(L/Lo)'] ))
        del Cool
    
    # PG cooling tracks (Althaus et al. 2009)
    if normal_mass_model == 'PG' and spec_type == 'He':
        for mass in ['0514','0530','0542','0565','0584','0609','0664','0741',
                     '0869']:
            Cool = Table.read('models/tracksPG-DB/db-pg-' + mass + '.trk.t0.11',
                              format='ascii', comment='#')
            Cool = Cool[Cool['age[Myr]'] > 0][::1] # (Cool['Log(Teff)'] > logteff_min) * (Cool['Log(Teff)'] < logteff_max)
            mass_array  = np.concatenate(( mass_array, np.ones(len(Cool))*int(mass)/1000 ))
            logg        = np.concatenate(( logg, Cool['Log(grav)'] ))
            age         = np.concatenate(( age, Cool['age[Myr]'] * 1e6 +
                                                (IFMR(int(mass)/1000))**(t_index) * 1e10 ))
            age_cool    = np.concatenate(( age_cool, Cool['age[Myr]'] * 1e6 ))
            logteff     = np.concatenate(( logteff, Cool['Log(Teff)'] ))
            Mbol        = np.concatenate(( Mbol, 4.75 - 2.5 * Cool['Log(L)'] ))
            del Cool
    
    # Ultra-massive ONe model (Camisassa et al. 2019)
    if high_mass_model == 'ONe':
        for mass in ['110','116','122','129']:
            Cool = Table.read('models/ONeWDs/' + mass + '_' + spec_suffix2 +
                              '.trk', format='ascii') 
            Cool = Cool[::10] # (Cool['LOG(TEFF)'] > logteff_min) * (Cool['LOG(TEFF)'] < logteff_max)
            #Cool.sort('Log(edad/Myr)')
            mass_array  = np.concatenate(( mass_array, np.ones(len(Cool)) * int(mass)/100 ))
            logg        = np.concatenate(( logg, Cool['Log(grav)'] ))
            age         = np.concatenate(( age, (10**Cool['Log(edad/Myr)'] -
                                                 10**Cool['Log(edad/Myr)'][0]) * 1e6 +
                                                (IFMR(int(mass)/100))**(t_index) * 1e10 ))
            age_cool    = np.concatenate(( age_cool, (10**Cool['Log(edad/Myr)'] -
                                                      10**Cool['Log(edad/Myr)'][0]) * 1e6 ))
            logteff     = np.concatenate(( logteff, Cool['LOG(TEFF)'] ))
            Mbol        = np.concatenate(( Mbol, 4.75 - 2.5 * Cool['LOG(L)'] ))
            # additional
            logT_c      = Cool['T_c'] + 6
            logrho_c    = Cool['Ro_c']
            XH_c        = Cool['Hc']
            XHe_c       = Cool['Hec']
            mass_accurate= Cool['Masa']
            logL_nu     = Cool['Log(Lnu)']
            logMH       = Cool['LogMHtot']
            logHeBuf    = Cool['LogHeBuf']
            logr        = np.log10(Cool['R/R_sun'])
            L_LH        = Cool['L.H.[erg/s)]'] / 3.828e33
            L_PS        = Cool['Sep.Fase[erg/s]'] / 3.828e33
            #/M^dot, Masa_HFC, Masa_HeFC 
            del Cool
    
    # massive MESA model (Lauffer et al. 2019)
    if high_mass_model == 'MESA': 
        if spec_type == 'He':
#             mesa_masslist = ['1.0124','1.019','1.0241','1.0358','1.0645','1.0877',
#                              '1.1102','1.1254','1.1313','1.1322','1.1466','1.151',
#                              '1.2163','1.22','1.2671','1.3075']
#                             ['1.0124','1.019','1.0241','1.0358','1.0645','1.0877',
#                              '1.1102','1.125','1.1309','1.1322','1.1466','1.151',
#                              '1.2163','1.22','1.2671','1.3075']
            mesa_masslist = ['1.0124','1.0645',
                             '1.1102','1.151',
                             '1.2163','1.2671','1.3075']
        elif spec_type == 'H':
            mesa_masslist = ['1.0124','1.0645',
                             '1.1102','1.151',
                             '1.2163','1.2671','1.3075']
        for mass in mesa_masslist:
            Cool = Table.read('models/MESA_model/' + spec_suffix3 + '_atm-M' +
                              mass + '.dat',
                              format='csv', header_start=1, data_start=2)
            dn = 70
            if float(mass) > 1.2:
                dn = 120
            if float(mass) < 1.05:
                dn = 10
            Cool = Cool[::dn] # [(Cool['# log Teff [K]'] > logteff_min) * (Cool['# log Teff [K]'] < logteff_max)]
            mass_array  = np.concatenate(( mass_array, np.ones(len(Cool)) * float(mass) ))
            logg        = np.concatenate(( logg, Cool['log g [cm/s^2]'] ))
            age         = np.concatenate(( age, Cool['total age [Gyr]'] * 1e9 ))
            age_cool    = np.concatenate(( age_cool, Cool['cooling age [Gyr]'] * 1e9))
            logteff     = np.concatenate(( logteff, Cool['# log Teff [K]'] ))
            Mbol        = np.concatenate(( Mbol, 4.75 - 2.5 * Cool['log L/Lsun'] ))
            # additional
            mass_accurate=Cool['mass [Msun]']
            logr        = Cool['log radius [Rsun]']
            del Cool
    
    select = ~np.isnan(mass_array + logg + age + age_cool + logteff + Mbol) * \
             (age_cool > 1e3)
    
    return mass_array[select], logg[select], age[select]*1e-9, age_cool[select]*1e-9, \
           logteff[select], Mbol[select]


def interp_HR_to_para(color, Mag, WD_para,
                      HR_grid=(-0.6, 1.5, 0.002, 8, 18, 0.01),
                      interp_type='linear'):
    """
    Interpolate the mapping of HR coordinate --> WD_para, based on the data 
    points from many cooling tracks read from a model, and get the value of a
    WD parameter on a grid of HR coordinates. The argument 'HR_grid' sets the
    range on the HR diagram for these interpolations.
    
    Args:
        color:          1d-array. 
                        The color index
        Mag:            1d-array. 
                        The absolute magnitude
        WD_para:        1d-array. 
                        The target parameter for mapping HR --> WD_para
        HR_grid:        (xmin, xmax, dx, ymin, ymax, dy). *Optional*
                        The grid information of the H-R diagram coordinates 
                        color and Mag
        interp_type:    String. {'linear', 'cubic'}. *Optional*
                        Linear is better for this purpose.
                  
    Returns:
        grid_para:      2d-array. 
                        The values of z on the grid of HR diagram
        HR_to_para:     Function. 
                        The mapping of HR coordinate --> WD_para
    
    """
    # define the grid of H-R diagram
    grid_x, grid_y = np.mgrid[HR_grid[0]:HR_grid[1]:HR_grid[2],
                              HR_grid[3]:HR_grid[4]:HR_grid[5]]
    
    # select only not-NaN data points
    selected    = ~np.isnan(color + Mag + WD_para) * (Mag > HR_grid[3]) * \
                  (Mag < HR_grid[4]) * (color > HR_grid[0]) * (color < HR_grid[1])
    
    # get the value of z on a H-R diagram grid and the interpolated mapping
    grid_para   = griddata(np.array((color[selected], Mag[selected])).T, 
                           WD_para[selected], (grid_x, grid_y), 
                           method=interp_type,
                           rescale=True)
    HR_to_para  = interpolate_2d(color[selected], Mag[selected], 
                                 WD_para[selected],
                                 interp_type)
    
    # return both the grid data and interpolated mapping
    return grid_para, HR_to_para


def interp_xy_z(x, y, z, xy_grid, interp_type='linear'):
    """Interpolate the mapping (x, y) --> z
    
    Interpolate the mapping (x, y) --> z, based on a series of x, y, and z
    values, and get the value of z on the grid of (x,y) coordinates. This
    function is a generalized version of HR_to_para, allowing any x and y 
    values.
    
    Args:
        x:              1d-array. The x in the mapping (x, y) --> z
        y:              1d-array. The y in the mapping (x, y) --> z
        z:              1d-array. The target parameter for mapping (x, y) --> z
        xy_grid:        (xmin, xmax, dx, ymin, ymax, dy).
                        The grid information of x and y
        interp_type:    String. {'linear', 'cubic'}. *Optional*
                        Linear is usually better for interpolating cooling 
                        tracks.
    
    Returns:
        grid_z:         2d-array. The values of z on the grid of (x, y)
        xy_to_z:        Function. The mapping of (x, y) --> z
    
    """
    # define the grid of (x,y)
    grid_x, grid_y = np.mgrid[xy_grid[0]:xy_grid[1]:xy_grid[2],
                              xy_grid[3]:xy_grid[4]:xy_grid[5]]
    
    # select only not-NaN data points
    selected = ~np.isnan(x + y + z)
    
    # get the value of z on a (x,y) grid and the interpolated mapping
    grid_z      = griddata(np.array((x[selected], y[selected])).T,
                           z[selected], (grid_x, grid_y), method=interp_type,
                           rescale=True)
    xy_to_z     = interpolate_2d(x[selected], y[selected], z[selected],
                                 interp_type)
    
    # return both the grid data and interpolated mapping
    return grid_z, xy_to_z
  

def interp_xy_z_func(x, y, z, interp_type='linear'):
    """Interpolate the mapping (x, y) --> z
    
    Interpolate the mapping (x, y) --> z, based on a series of x, y, and z
    values. This function is a generalized version of HR_to_para, allowing any
    x and y values, but does not calculate the grid values as HR_to_para and
    interp_xy_z do.
    
    Args:
        x:              1d-array. The Gaia color BP-RP
        y:              1d-array. The absolute magnitude of Gaia G band
        z:              1d-array. The target parameter for mapping (x, y) --> z
        interp_type:    String. {'linear', 'cubic'}. *Optional*
                        Linear is usually better for interpolating cooling 
                        tracks.
    
    Returns:
        xy_to_z:        Function. The mapping of (x, y) --> z
        
    """
    # select only not-NaN data points
    selected    = ~np.isnan(x + y + z)
    
    # get the interpolated mapping
    xy_to_z     = interpolate_2d(x[selected], y[selected], z[selected],
                                 interp_type)
    
    # return only the interpolated mapping
    return xy_to_z


#-------------------------------------------------------------------------------  
#
#   Define the main function that reads a set of cooling tracks and generate 
#   useful mappings
#
#-------------------------------------------------------------------------------   


def load_model(low_mass_model, normal_mass_model, high_mass_model, spec_type,
               HR_bands=('bp-rp', 'G'),
               HR_grid=(-0.6, 1.5, 0.002, 8, 18, 0.01),
               logteff_logg_grid=(3.5, 5.1, 0.01, 6.5, 9.6, 0.01),
               interp_type_atm='linear', interp_type='linear',
               for_comparison=False):
    """ Load a set of cooling tracks and interpolate the HR diagram mapping
    
    This function reads a set of cooling tracks assigned by the user and returns
    many useful grid-values for plotting the contour of WD parameters on the
    Gaia H--R diagram and functions for mapping between Gaia photometry and 
    WD parameters.
    
    For other mappings not included in the output, the user can generate the
    interpolated grid values and mapping function based on the cooling-track 
    data points and atmosphere models that are also provided in the output.
    E.g., for the mapping (mass, logteff) --> cooling age,
    (```)
    import WD_models
    model = WD_models.load_model('', 'b', 'b', 'DA_thick')
    m_logteff_to_agecool = WD_models.interp_xy_z_func(model['mass_array'],
                                                      model['logteff'],
                                                      model['age_cool']
                                                     )
    m_logteff_to_agecool(1.1, np.log10(10000)) # calling the function
    (```)
    
    Args:
        low_mass_model:     String. 
                            Specifying the cooling model used for low-mass WDs
                            (<~0.5Msun). Its value should be one of the
                            following: 
            ''                              no low-mass model will be read
            'Fontaine2001' or 'f'           the think-hydrogen CO WD model in 
                                            http://www.astro.umontreal.ca/~bergeron/CoolingModels/
            'Fontaine2001_thin' or 'ft'     the thin-hydrogen CO WD model in 
                                            http://www.astro.umontreal.ca/~bergeron/CoolingModels/
        normal_mass_model:  String. 
                            Specifying the cooling model used for normal-mass
                            WDs (about 0.5~1.0Msun). Its value should be one of
                            the following:
            ''                              no normal-mass model will be read
            'Fontaine2001' or 'f'           the think-hydrogen CO WD model in 
                                            http://www.astro.umontreal.ca/~bergeron/CoolingModels/
            'Fontaine2001_thin' or 'ft'     the thin-hydrogen CO WD model in 
                                            http://www.astro.umontreal.ca/~bergeron/CoolingModels/
            'Althaus2010_001' or 'a001'     Z=0.01, only for DA, http://evolgroup.fcaglp.unlp.edu.ar/TRACKS/tracks_cocore.html
            'Althaus2010_0001' or 'a0001'   Z=0.001, only for DA, http://evolgroup.fcaglp.unlp.edu.ar/TRACKS/tracks_cocore.html
            'Camisassa2017' or 'c'          only for DB, http://evolgroup.fcaglp.unlp.edu.ar/TRACKS/tracks_DODB.html
            'BaSTI' or 'b'                  with phase separation, Salaris et al. 2010, http://basti.oa-teramo.inaf.it
            'BaSTI_nosep' or 'bn'           no phase separation, Salaris et al. 2010, http://basti.oa-teramo.inaf.it
            'PG'                            only for DB
        high_mass_model:    String. 
                            Specifying the cooling model used for high-mass WDs
                            (>~1.0Msun). Should be one of the following: 
            ''                              no high-mass model will be read
            'Fontaine2001' or 'f'           the think-hydrogen CO WD model in 
                                            http://www.astro.umontreal.ca/~bergeron/CoolingModels/
            'Fontaine2001_thin' or 'ft'     the thin-hydrogen CO WD model in 
                                            http://www.astro.umontreal.ca/~bergeron/CoolingModels/
            'ONe' or 'o'                    Camisassa et al. 2019, http://evolgroup.fcaglp.unlp.edu.ar/TRACKS/ultramassive.html
            'MESA' or 'm'                   Lauffer et al. 2019
            'BaSTI' or 'b'                  with phase separation, Salaris et al. 2010, http://basti.oa-teramo.inaf.it
            'BaSTI_nosep' or 'bn'           no phase separation, Salaris et al. 2010, http://basti.oa-teramo.inaf.it
        spec_type:          String. 
                            Specifying the atmosphere composition. Its value 
                            should be one of the following:
            'H'                             pure-hydrogen atmosphere
            'He'                            pure-helium atmosphere
        HR_bands:          (String, String). *Optional*
            The passbands for the color and absolute magnitude on the H--R
            diagram. It can be any combination from the following bands:
                G, bp, rp (Gaia), u, g, r, i, z (SDSS), U, B, V, R, I, J,
                H, K (Johnson).
            Color should be in the format of:
                'bp-rp', 'G-bp', 'U-B', 'u-g', 'G-i', etc. 
            Absolute magnitude should be in the format of:
                'G', 'bp', 'U', 'u', etc. 
        HR_grid:           (xmin, xmax, dx, ymin, ymax, dy). *Optional*
            The grid information of the H-R diagram coordinates BP-RP and G.
        logteff_logg_grid: (xmin, xmax, dx, ymin, ymax, dy). *Optional*
            The grid information of the logteff--logg coordinates for 
            the table interpolation of the atmosphere synthetic colors. Since
            the DA cooling track have a turning-back of color index below around
            3500 K, the user should set logteff_logg_grid[0] >= 3.5.
        interp_type_atm:    String. {'linear', 'cubic'}
        interp_type:        String. {'linear', 'cubic'}
            Linear is better for interpolating WD cooling tracks.
        for_comparison:     Bool. *Optional*
            If true, cooling tracks with very similar masses from different 
            models will be used, which might lead to strange result of
            interpolation. 
            E.g., the Fontaine2001 model has m_WD = [..., 0.95, 1.00, ...], and
            the MESA model has m_WD = [1.0124, 1.019, ...]. If true, the 
            Fontaine2001 1.00Msun cooling track will be used; if false, it will
            not be used because it is too close to the MESA 1.0124Msun track.
        
    Returns:
        A Dictionary.
        It contains the atmosphere grids and mapping, cooling-track data points,
        and parameter mappings based on the cooling tracks. 
        The keys of this dictionary are:
            interpolation results:
        ========================================================================
          category   | interpolated values on a grid | interpolated mapping
          var. type  |     2d-array                  |     Function
        ========================================================================
           atm.      | 'grid_logteff_logg_to_BC'     | 'logteff_logg_to_BC'   (bolometric correction)
                     | 'grid_logteff_logg_to_color'  | 'logteff_logg_to_color'
        ------------------------------------------------------------------------
         HR -->      | 'grid_HR_to_mass'             | 'HR_to_mass'
         WD para.    | 'grid_HR_to_logg'             | 'HR_to_logg'
                     | 'grid_HR_to_age'              | 'HR_to_age'
                     | 'grid_HR_to_age_cool'         | 'HR_to_age_cool'
                     | 'grid_HR_to_logteff'          | 'HR_to_logteff'
                     | 'grid_HR_to_Mbol'             | 'HR_to_Mbol'
                     | 'grid_HR_to_cool_rate^-1'     | 'HR_to_cool_rate^-1'
        ------------------------------------------------------------------------
         others      |                               | 'm_agecool_to_color'
                     |                               | 'm_agecool_to_Mag'
        ======================================================================== 
            cooling-track data points:
        'mass_array':   1d-array. The mass of WD in unit of solar mass. I only 
                        read one value for a cooling track, not tracking the 
                        mass change.
        'logg':         1d-array. in cm/s^2
        'age':          1d-array. The total age of the WD in Gyr. Some are read
                        directly from the cooling tracks, but others are 
                        calculated by assuming an initial--final mass relation
                        (IFMR) of the WD and adding the rough main-sequence age
                        to the cooling age.
        'age_cool':     1d-array. The cooling age of the WD in Gyr.
        'logteff':      1d-array. The logarithm effective temperature of the WD
                        in Kelvin (K).
        'Mbol':         1d-array. The absolute bolometric magnitude of the WD. 
                        Many are converted from the log(L/Lsun) or log(L), where
                        I adopt:
                                Mbol_sun = 4.75,
                                Lsun = 3.828e33 erg/s.
        'cool_rate^-1': 1d-array. The reciprocal of cooling rate dt / d(bp-rp),
                        in Gyr/mag.
        'Mag':          1d-array. The absolute magnitude of the chosen passband
                        converted from the atmosphere interpolation.
        'color':        1d-array. The chosen color index, converted from the
                        atmosphere interpolation.
                        
    """
    # define some alias of model names
    if low_mass_model == 'f':
        low_mass_model = 'Fontaine2001'
    if low_mass_model == 'ft':
        low_mass_model = 'Fontaine2001_thin'
    
    if normal_mass_model == 'a001':
        normal_mass_model = 'Althaus2010_001'
    if normal_mass_model == 'a0001':
        normal_mass_model = 'Althaus2010_0001'
    if normal_mass_model == 'f':
        normal_mass_model = 'Fontaine2001'
    if normal_mass_model == 'ft':
        normal_mass_model = 'Fontaine2001_thin'
    if normal_mass_model == 'c':
        normal_mass_model = 'Camisassa2017'
    if normal_mass_model == 'b':
        normal_mass_model = 'BaSTI'
    if normal_mass_model == 'bn':
        normal_mass_model = 'BaSTI_nosep'
    
    if high_mass_model == 'f':
        high_mass_model = 'Fontaine2001'
    if high_mass_model == 'ft':
        high_mass_model = 'Fontaine2001_thin'
    if high_mass_model == 'b':
        high_mass_model = 'BaSTI'
    if high_mass_model == 'bn':
        high_mass_model = 'BaSTI_nosep'
    if high_mass_model == 'm':
        high_mass_model = 'MESA'
    if high_mass_model == 'o':
        high_mass_model = 'ONe'
    
    model_names = ['', 'Fontaine2001', 'Fontaine2001_thin', 'Althaus2010_001', 
                   'Althaus2010_0001', 'Camisassa2017', 'BaSTI', 'BaSTI_nosep',
                   'MESA', 'ONe',]
    if (low_mass_model not in model_names or 
        normal_mass_model not in model_names or
        high_mass_model not in model_names):
        print('please check the model names.')
    if spec_type not in ['H', 'He']:
        print('please enter either \'H\' or \'He\' for spec_type.')
    
    # make atmosphere grid and mapping: logteff, logg --> bp-rp,  G-Mbol
    grid_logteff_logg_to_color, logteff_logg_to_color = interp_atm(
        spec_type, HR_bands[0], 
        logteff_logg_grid=logteff_logg_grid,
        interp_type_atm=interp_type_atm)
    grid_logteff_logg_to_BC, logteff_logg_to_BC = interp_atm(
        spec_type, HR_bands[1] + '-Mbol', 
        logteff_logg_grid=logteff_logg_grid,
        interp_type_atm=interp_type_atm)


    # get for logg_func BaSTI models
    if 'BaSTI' in normal_mass_model or 'BaSTI' in high_mass_model:
        mass_array_Fontaine2001, logg_Fontaine2001, _, _, logteff_Fontaine2001, _\
                    = read_cooling_tracks('Fontaine2001',
                                          'Fontaine2001',
                                          'Fontaine2001',
                                          spec_type)
        logg_func   = interp_xy_z_func(x=logteff_Fontaine2001,
                                       y=mass_array_Fontaine2001,
                                       z=logg_Fontaine2001)
    else: 
        logg_func   = None
    
    # Open Evolutionary Tracks
    mass_array, logg, age, age_cool, logteff, Mbol \
                    = read_cooling_tracks(low_mass_model,
                                          normal_mass_model,
                                          high_mass_model,
                                          spec_type, logg_func, for_comparison)

    # Get Colour/Magnitude for Evolution Tracks
    Mag         = logteff_logg_to_BC(logteff, logg) + Mbol
    color       = logteff_logg_to_color(logteff, logg)
    
    # Calculate the Recipical of Cooling Rate (Cooling Time per BP-RP)
    k1        = (age_cool[1:-1] - age_cool[:-2]) / (color[1:-1] - color[:-2])
    k2        = (age_cool[2:] - age_cool[1:-1]) / (color[2:] - color[1:-1])
    k         = k1 + (color[1:-1] - color[:-2]) * (k1-k2) / (color[:-2]-color[2:])
    rate_inv  = np.concatenate(( np.array([1]), k , np.array([1]) ))
        
    # Get Parameters on HR Diagram
    grid_HR_to_mass, HR_to_mass         = interp_HR_to_para(color, Mag, mass_array, 
                                                            HR_grid, interp_type)
    grid_HR_to_logg, HR_to_logg         = interp_HR_to_para(color, Mag, logg, 
                                                            HR_grid, interp_type)
    grid_HR_to_age, HR_to_age           = interp_HR_to_para(color, Mag, age, 
                                                            HR_grid, interp_type)
    grid_HR_to_age_cool, HR_to_age_cool = interp_HR_to_para(color, Mag, age_cool, 
                                                            HR_grid, interp_type)
    grid_HR_to_logteff, HR_to_logteff   = interp_HR_to_para(color, Mag, logteff, 
                                                            HR_grid, interp_type)
    grid_HR_to_Mbol, HR_to_Mbol         = interp_HR_to_para(color, Mag, Mbol, 
                                                            HR_grid, interp_type)
    grid_HR_to_rate_inv, HR_to_rate_inv = interp_HR_to_para(color, Mag, rate_inv,
                                                            HR_grid, interp_type)
    # (mass, t_cool) --> bp-rp, G
    m_agecool_to_color                  = interp_xy_z_func(mass_array, age_cool,
                                                           color, interp_type)
    m_agecool_to_Mag                    = interp_xy_z_func(mass_array, age_cool,
                                                           Mag, interp_type)
    
    # Return a dictionary containing all the cooling track data points, 
    # interpolation functions and interpolation grids 
    return {'grid_logteff_logg_to_BC':grid_logteff_logg_to_BC,
            'logteff_logg_to_BC':logteff_logg_to_BC,
            'grid_logteff_logg_to_color':grid_logteff_logg_to_color,
            'logteff_logg_to_color':logteff_logg_to_color,
            'mass_array':mass_array, 'logg':logg, 'logteff':logteff,
            'age':age, 'age_cool':age_cool, 'cool_rate^-1':rate_inv,
            'Mbol':Mbol, 'Mag':Mag, 'color':color,
            'grid_HR_to_mass':grid_HR_to_mass, 'HR_to_mass':HR_to_mass,
            'grid_HR_to_logg':grid_HR_to_logg, 'HR_to_logg':HR_to_logg,
            'grid_HR_to_age':grid_HR_to_age, 'HR_to_age':HR_to_age,
            'grid_HR_to_age_cool':grid_HR_to_age_cool, 'HR_to_age_cool':HR_to_age_cool,
            'grid_HR_to_logteff':grid_HR_to_logteff, 'HR_to_logteff':HR_to_logteff,
            'grid_HR_to_Mbol':grid_HR_to_Mbol, 'HR_to_Mbol':HR_to_Mbol,
            'grid_HR_to_cool_rate^-1':grid_HR_to_rate_inv, 'HR_to_cool_rate^-1':HR_to_rate_inv,
            'm_agecool_to_color':m_agecool_to_color, 
            'm_agecool_to_Mag':m_agecool_to_Mag}
