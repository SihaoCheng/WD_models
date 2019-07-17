"""
This package will read different cooling models and interpolate the conversion 
mappings between HR diagram, Teff, Mbol, etc. The mappings are stored in 
dictionaries for each model. See the main function and the lines after its 
definition. This package also contains the functions to read a single cooling 
track.
"""


import matplotlib.pyplot as plt
import numpy as np

from astropy.table import Table
from scipy.interpolate import CloughTocher2DInterpolator, LinearNDInterpolator
from scipy.interpolate import griddata, interp1d


#-------------------------------------------------------------------------------   
#
#   Define some constants that will be used as parameters of the following 
#   functions
#
#-------------------------------------------------------------------------------


tmin = 3.5; tmax = 5.1; dt = 0.01
loggmin = 6.5; loggmax = 9.6; dlogg = 0.01
HR_grid             = (-0.6, 1.5, 0.002, 8, 16, 0.01) # bp_rp, G
interp_type         = 'linear'
interp_type_atm     = 'linear'
interp_bprp_factor  = 5


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
  

def interp_atm(spec_type, color, T_logg_grid=(3.5, 5.1, 0.01, 6.5, 9.6, 0.01), 
               interp_type_atm='linear'):
    """interpolate the mapping (logteff, logg) --> G, BP-RP, or G-Mbol 
    
    Args:
        spec_type:    string. 'DA_thick' or 'DA_thin' or 'DB'. 
                      See http://www.astro.umontreal.ca/~bergeron/CoolingModels/
        color:        string. 'G', 'BP-RP', or 'G-Mbol'. This is the target 
                      photometry of the mapping. 
        T_logg_grid:  in the form (xmin, xmax, dx, ymin, ymax, dy), 
                      corresponding to the grid of logTeff and logg.
    
    Returns:
        grid_z:       2darray. The value of photometry on a (logteff, logg) grid
        grid_z_func:  function. The interpolated mapping
    """
    logteff     = np.zeros(0)
    logg        = np.zeros(0)
    age         = np.zeros(0)
    mass_array  = np.zeros(0)
    G           = np.zeros(0)
    bp_rp       = np.zeros(0)
    Mbol        = np.zeros(0)
    
    # read the table for each mass
    for mass in ['0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0','1.2']:
        Atm_color = Table.read('models/Montreal_atm_grid/Table_Mass_' + mass +
                               '_'+spec_type, format='ascii')
        selected  = Atm_color['Teff'] > 3500
        Atm_color = Atm_color[selected]
        
        # read columns
        bp_rp   = np.concatenate(( bp_rp, 
                                   Atm_color['G_BP/R'] - Atm_color['G_RP/R'] ))
        G       = np.concatenate(( G, Atm_color['G/R'] ))
        Mbol    = np.concatenate(( Mbol, Atm_color['Mbol'] ))
        logteff = np.concatenate(( logteff, np.log10(Atm_color['Teff']) ))
        logg    = np.concatenate(( logg, Atm_color['logg'] ))
        age     = np.concatenate(( age, Atm_color['Age'] ))
        
    # read the table for all logg
    Atm_color = Table.read('models/Montreal_atm_grid/Table_'+spec_type,
                           format='ascii')
    selected  = Atm_color['Teff'] > 3500
    Atm_color = Atm_color[selected]
    
    # read columns
    bp_rp   = np.concatenate(( bp_rp, 
                               Atm_color['G_BP/R'] - Atm_color['G_RP/R'] ))
    G       = np.concatenate(( G, Atm_color['G/R'] ))
    Mbol    = np.concatenate(( Mbol, Atm_color['Mbol'] ))
    logteff = np.concatenate(( logteff,np.log10(Atm_color['Teff']) ))
    logg    = np.concatenate(( logg, Atm_color['logg'] ))
    age     = np.concatenate(( age, Atm_color['Age'] ))        
    
    grid_x, grid_y = np.mgrid[T_logg_grid[0]:T_logg_grid[1]:T_logg_grid[2],
                              T_logg_grid[3]:T_logg_grid[4]:T_logg_grid[5]]
    
    # define the interpolation of mapping
    def interp(x,y,z):
        grid_z      = griddata(np.array((x,y)).T, z, (grid_x, grid_y),
                               method=interp_type_atm)
        grid_z_func = interpolate_2d(x, y, z, interp_type_atm)
        return grid_z, grid_z_func
    
    if color == 'G':
        return interp(logteff, logg, G)
    if color == 'bp_rp':
        return interp(logteff, logg, bp_rp)
    if color == 'G_Mbol':
        return interp(logteff, logg, G-Mbol)


def read_cooling_tracks(normal_mass_model, high_mass_model, spec_type, 
                          logg_func=None, for_comparison=False):
    """ Read a set of cooling tracks
    
    This function reads the cooling models and stack together the data points
    of mass, logg, age, age_cool, logteff, and Mbol from different cooling
    tracks.
    
    Args:
        normal_mass_model:  string. One of the following: 
            'Fontaine2001' or 'f'           http://www.astro.umontreal.ca/~bergeron/CoolingModels/
            'Althaus2010_001' or 'a001'     Z=0.01, only for DA, http://evolgroup.fcaglp.unlp.edu.ar/TRACKS/tracks_cocore.html
            'Althaus2010_0001' or 'a0001'   Z=0.001, only for DA, http://evolgroup.fcaglp.unlp.edu.ar/TRACKS/tracks_cocore.html
            'Camisassa2017' or 'c'          only for DB, http://evolgroup.fcaglp.unlp.edu.ar/TRACKS/tracks_DODB.html
            'BaSTI' or 'b'                  with phase separation, Salaris et al. 2010, http://basti.oa-teramo.inaf.it
            'BaSTI_nosep' or 'bn'           no phase separation, Salaris et al. 2010, http://basti.oa-teramo.inaf.it
            'PG'                            only for DB
        high_mass_model:    string. One of the following: 
            'Fontaine2001' or 'f'           http://www.astro.umontreal.ca/~bergeron/CoolingModels/
            'ONe' or 'o'                    Camisassa et al. 2019, http://evolgroup.fcaglp.unlp.edu.ar/TRACKS/ultramassive.html
            'MESA' or 'm'                   Lauffer et al. 2019
            'BaSTI' or 'b'                  with phase separation, Salaris et al. 2010, http://basti.oa-teramo.inaf.it
            'BaSTI_nosep' or 'bn'           no phase separation, Salaris et al. 2010, http://basti.oa-teramo.inaf.it
        spec_type:          string. One of the following:
            'DA_thick'
            'DA_thin'
            'DB'
        logg_func:          Function. 
            This is a function for (logteff, mass) --> logg. It is necessary 
            only for BaSTI models, because the BaSTI models do not directly 
            provide log g information.
        for_comparison:     Bool. 
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
        age:        1d-array. The total age of the WD in yr. Some are read
                    directly from the cooling tracks, but others are calculated
                    by assuming an initial--final mass relation (IFMR) of the WD
                    and adding the rough main-sequence age to the cooling age.
        age_cool:   1d-array. The cooling age of the WD in yr.
        logteff:    1d-array. The logarithm effective temperature of the WD in
                    Kelvin (K).
        Mbol:       1d-array. The absolute bolometric magnitude of the WD. Many
                    are converted from the log(L/Lsun) or log(L), where I adopt:
                        Mbol_sun = 4.75
                        Lsun = 3.828e33 erg/s
    
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
    
    if high_mass_model == 'Fontaine2001':
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
    
    # define atmosphere
    if spec_type == 'DB':
        spec_suffix = '0210'; spec_suffix2 = 'DB'; spec_suffix3 = 'He'
    else:
        spec_suffix = '0204'; spec_suffix2 = 'DA'; spec_suffix3 = 'H'
    
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
    
    # read data from cooling models
    # Fontaine et al. 2001
    for mass in ['020','030','040','050','060','070','080','090','095','100',
                 '105','110','115','120','125','130']:
        if (int(mass)/100 < mass_separation_1 or
            (normal_mass_model == 'Fontaine2001' and
             int(mass)/100 < mass_separation_2 ) or
            (high_mass_model == 'Fontaine2001' and
             int(mass)/100 > mass_separation_2 )
           ):
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
                logg_temp.append(     float(text[line*l_line+22:
                                                 line*l_line+35]) )
                age_temp.append(      float(text[line*l_line+48:
                                                 line*l_line+63]) +
                                      (IFMR(int(mass)/100))**(t_index) * 1e10 )
                age_cool_temp.append( float(text[line*l_line+48:
                                                 line*l_line+63]) )
                Mbol_temp.append(     4.75 - 
                                      2.5 * np.log10(float(text[line*l_line+64:
                                                                line*l_line+76]
                                                          ) / 3.828e33) )
            mass_array  = np.concatenate(( mass_array, np.ones(len(age_temp)) * int(mass)/100 ))
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
    if 'Althaus2010_' in normal_mass_model and 'DA' in spec_type:
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
            Cool = Cool[(Cool['log(TEFF)'] > tmin) *
                        (Cool['log(TEFF)'] < tmax)][::1]
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
    if normal_mass_model == 'Camisassa2017' and spec_type == 'DB':
        for mass in ['051','054','058','066','074','087','100']:
            if int(mass)/100 < mass_separation_2:
                Cool = Table.read('models/Camisassa_2017_DB_CO/Z002/' + mass +
                                  'DB.trk', format='ascii') 
                dn = 1
                if int(mass)/100 > 0.95:
                    dn = 50
                Cool = Cool[(Cool['LOG(TEFF)'] > tmin) *
                            (Cool['LOG(TEFF)'] < tmax)][::dn]
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
        normal_mass_use_BaSTI = (normal_mass_model == 'BaSTI' or 
                                 normal_mass_model == 'BaSTI_nosep'
                                )
        high_mass_use_BaSTI = (high_mass_model == 'BaSTI' or 
                               high_mass_model == 'BaSTI_nosep'
                              )
        sep = 'sep'
        if normal_mass_use_BaSTI and int(mass)/100 < mass_separation_2:
            if 'nosep' in normal_mass_model:
                sep = 'nosep'
            Cool = Table.read('models/BaSTI/COOL' + mass + 'BaSTIfinale' + 
                   spec_suffix2 + sep +'.sdss', format='ascii')
        elif high_mass_use_BaSTI and int(mass)/100 > mass_separation_2:
            if 'nosep' in high_mass_model:
                sep = 'nosep'
            Cool = Table.read('models/BaSTI/COOL' + mass + 'BaSTIfinale' + 
                   spec_suffix2 + sep +'.sdss', format='ascii')
        else:
            continue
        dn = 1
        if int(mass)/100 > 1.05:
            dn = 5
        Cool = Cool[(Cool['log(Teff)'] > tmin) *
                    (Cool['log(Teff)'] < tmax)][::dn]
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
    if normal_mass_model == 'PG' and spec_type == 'DB':
        for mass in ['0514','0530','0542','0565','0584','0609','0664','0741',
                     '0869']:
            Cool = Table.read('models/tracksPG-DB/db-pg-' + mass + '.trk.t0.11',
                              format='ascii', comment='#')
            Cool = Cool[(Cool['Log(Teff)'] > tmin) *
                        (Cool['Log(Teff)'] < tmax) *
                        (Cool['age[Myr]'] > 0)]
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
            Cool = Cool[(Cool['LOG(TEFF)'] > tmin) *
                        (Cool['LOG(TEFF)'] < tmax)][::10]
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
        if spec_suffix3 == 'He':
#             mesa_masslist = ['1.0124','1.019','1.0241','1.0358','1.0645','1.0877',
#                              '1.1102','1.1254','1.1313','1.1322','1.1466','1.151',
#                              '1.2163','1.22','1.2671','1.3075']
#                             ['1.0124','1.019','1.0241','1.0358','1.0645','1.0877',
#                              '1.1102','1.125','1.1309','1.1322','1.1466','1.151',
#                              '1.2163','1.22','1.2671','1.3075']
            mesa_masslist = ['1.0124','1.0645',
                             '1.1102','1.151',
                             '1.2163','1.2671','1.3075']
        else:
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
            Cool = Cool[(Cool['# log Teff [K]'] > tmin) *
                        (Cool['# log Teff [K]'] < tmax)][::dn]
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
    
    return mass_array[select], logg[select], age[select], age_cool[select], \
           logteff[select], Mbol[select]


def interp_HR_to_para(bp_rp, G, para, age, 
                      HR_grid=(-0.6, 1.5, 0.002, 10, 15, 0.01),
                      interp_type='linear'):
    """
    Interpolate the mapping of (BR-RP, G) --> para, based on the data points 
    from many cooling tracks read from a model, and get the value of z on the
    grid of H-R coordinates. We set select only G < 16 and G > 8 to avoid the 
    turning of DA cooling track which leads to multi-value mapping.
    
    Args:
        bp_rp:      1d-array. The Gaia color BP-RP
        G:          1d-array. The absolute magnitude of Gaia G band
        para:       1d-array. The target parameter for mapping (BP-RP, G) --> para
        HR_grid:    in the form of (xmin, xmax, dx, ymin, ymax, dy), the grid 
                    information of the H-R diagram coordinates BP-RP and G
        age:        1d-array. The WD age. Only used for the purpose of selecting 
                    non-NaN data points.
                  
    Returns:
        grid_z:     2d-array. The values of z on the grid of HR diagram
        HR_to_z:    Function. The mapping of (BP-RP, G) --> z
    
    """
    # define the grid of H-R diagram
    grid_x, grid_y = np.mgrid[HR_grid[0]:HR_grid[1]:HR_grid[2],
                              HR_grid[3]:HR_grid[4]:HR_grid[5]]
    grid_x *= interp_bprp_factor
    
    # select only not-NaN data points
    selected    = ~np.isnan(bp_rp + G + age + para) * (G < 16) * (G > 8)
    
    # get the value of z on a H-R diagram grid and the interpolated mapping
    grid_para   = griddata(np.array((bp_rp[selected]*interp_bprp_factor,
                                     G[selected])).T, 
                           para[selected], (grid_x, grid_y), method=interp_type)
    HR_to_para  = interpolate_2d(bp_rp[selected], G[selected], para[selected],
                                 interp_type)
    
    # return both the grid data and interpolated mapping
    return grid_para, HR_to_para


def interp_xy_z(x, y, z, xy_grid, xfactor=1, interp_type='linear'):
    """Interpolate the mapping (x, y) --> z
    
    Interpolate the mapping (x, y) --> z, based on a series of x, y, and z
    values, and get the value of z on the grid of (x,y) coordinates. This
    function is a generalized version of HR_to_para, allowing any x and y 
    values.
    
    Args:
        x:          1d-array. The x in the mapping (x, y) --> z
        y:          1d-array. The y in the mapping (x, y) --> z
        z:          1d-array. The target parameter for mapping (x, y) --> z
        xy_grid:    in the form of (xmin, xmax, dx, ymin, ymax, dy), the grid 
                    information of x and y
        xfactor:    Number. For balancing the interval of interpolation between
                    x and y.
    
    Returns:
        grid_z:     2d-array. The values of z on the grid of (x, y)
        xy_to_z:    Function. The mapping of (x, y) --> z
    
    """
    # define the grid of (x,y)
    grid_x, grid_y = np.mgrid[xy_grid[0]:xy_grid[1]:xy_grid[2],
                              xy_grid[3]:xy_grid[4]:xy_grid[5]]
    grid_x *= xfactor
    
    # select only not-NaN data points
    selected = ~np.isnan(x + y + z)
    
    # get the value of z on a (x,y) grid and the interpolated mapping
    grid_z      = griddata(np.array((x[selected]*xfactor, y[selected])).T,
                           z[selected], (grid_x, grid_y), method=interp_type)
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


def load_model(normal_mass_model, high_mass_model, spec_type, 
               interp_type_atm='linear', interp_type='linear'):
    """ Load a set of cooling tracks and interpolate the mapping to HR diagram
    
    This function is the main function of the WD_models package:
    First, it reads the table of synthetic colors 
    First, it reads the mass, logg, total age (if it exists), cooling age,
    logteff, and absolute bolometric magnitude Mbol from white dwarf cooling
    models with the function 'read_cooling_tracks';
    Then, it interpolates the mapping between parameters such as logg, teffGaia photometry
    
    Args:
        normal_mass_model:  string. One of the following: 
            'Fontaine2001' or 'f'           http://www.astro.umontreal.ca/~bergeron/CoolingModels/
            'Althaus2010_001' or 'a001'     Z=0.01, only for DA, http://evolgroup.fcaglp.unlp.edu.ar/TRACKS/tracks_cocore.html
            'Althaus2010_0001' or 'a0001'   Z=0.001, only for DA, http://evolgroup.fcaglp.unlp.edu.ar/TRACKS/tracks_cocore.html
            'Camisassa2017' or 'c'          only for DB, http://evolgroup.fcaglp.unlp.edu.ar/TRACKS/tracks_DODB.html
            'BaSTI' or 'b'                  with phase separation, Salaris et al. 2010, http://basti.oa-teramo.inaf.it
            'BaSTI_nosep' or 'bn'           no phase separation, Salaris et al. 2010, http://basti.oa-teramo.inaf.it
            'PG'                            only for DB
        high_mass_model:    string. One of the following: 
            'Fontaine2001' or 'f'           http://www.astro.umontreal.ca/~bergeron/CoolingModels/
            'ONe' or 'o'                    Camisassa et al. 2019, http://evolgroup.fcaglp.unlp.edu.ar/TRACKS/ultramassive.html
            'MESA' or 'm'                   Lauffer et al. 2019
            'BaSTI' or 'b'                  with phase separation, Salaris et al. 2010, http://basti.oa-teramo.inaf.it
            'BaSTI_nosep' or 'bn'           no phase separation, Salaris et al. 2010, http://basti.oa-teramo.inaf.it
        spec_type:          string. One of the following:
            'DA_thick'
            'DA_thin'
            'DB'
        logg_func:          Function. 
            This is a function for (logteff, mass) --> logg. It is necessary 
            only for BaSTI models, because the BaSTI models do not directly 
            provide log g information.
        for_comparison:     Bool. 
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
        age:        1d-array. The total age of the WD in yr. Some are read
                    directly from the cooling tracks, but others are calculated
                    by assuming an initial--final mass relation (IFMR) of the WD
                    and adding the rough main-sequence age to the cooling age.
        age_cool:   1d-array. The cooling age of the WD in yr.
        logteff:    1d-array. The logarithm effective temperature of the WD in
                    Kelvin (K).
        Mbol:       1d-array. The absolute bolometric magnitude of the WD. Many
                    are converted from the log(L/Lsun) or log(L), where I adopt:
                        Mbol_sun = 4.75
                        Lsun = 3.828e33 erg/s
                        
    """
    
    # define some alias of model names
    if normal_mass_model == 'a001':
        normal_mass_model = 'Althaus_001'
    if normal_mass_model == 'a0001':
        normal_mass_model = 'Althaus_0001'
    if normal_mass_model == 'f':
        normal_mass_model = 'Fontaine2001'
    if normal_mass_model == 'c':
        normal_mass_model = 'Camisassa2017'
    if normal_mass_model == 'b':
        normal_mass_model = 'BaSTI'
    if normal_mass_model == 'bn':
        normal_mass_model = 'BaSTI_nosep'
    
    if high_mass_model == 'f':
        high_mass_model = 'Fontaine2001'
    if high_mass_model == 'b':
        high_mass_model = 'BaSTI'
    if high_mass_model == 'bn':
        high_mass_model = 'BaSTI_nosep'
    if high_mass_model == 'm':
        high_mass_model = 'MESA'
    if high_mass_model == 'o':
        high_mass_model = 'ONe'
    
    # make atmosphere grid and mapping: logteff, logg --> bp-rp,  G-Mbol
    grid_logteff_logg_to_G_Mbol, logteff_logg_to_G_Mbol = interp_atm(
        spec_type, 'G_Mbol', 
        T_logg_grid=(tmin,tmax,dt,loggmin,loggmax,dlogg),
        interp_type_atm=interp_type_atm)
    grid_logteff_logg_to_bp_rp, logteff_logg_to_bp_rp = interp_atm(
        spec_type, 'bp_rp', 
        T_logg_grid=(tmin,tmax,dt,loggmin,loggmax,dlogg),
        interp_type_atm=interp_type_atm)
    
    
    # get for logg_func BaSTI models
    if 'BaSTI' in normal_mass_model or 'BaSTI' in high_mass_model:
        mass_array_Fontaine2001, logg_Fontaine2001, _, _, logteff_Fontaine2001, _\
                    = read_cooling_tracks('Fontaine2001',
                                          'Fontaine2001',
                                          spec_type)
        logg_func   = interp_xy_z_func(x=logteff_Fontaine2001,
                                       y=mass_array_Fontaine2001,
                                       z=logg_Fontaine2001)
    else: 
        logg_func   = None
        
    
    # Open Evolution Tracks
    mass_array, logg, age, age_cool, logteff, Mbol \
                    = read_cooling_tracks(normal_mass_model,
                                          high_mass_model,
                                          spec_type, logg_func)
    

    # Get Colour/Magnitude for Evolution Tracks
    G       = logteff_logg_to_G_Mbol(logteff, logg) + Mbol
    bp_rp   = logteff_logg_to_bp_rp(logteff, logg)
    
    
    # Calculate the Recipical of Cooling Rate (Cooling Time per BP-RP)
    k1          = (age_cool[1:-1] - age_cool[:-2]) / (bp_rp[1:-1] - bp_rp[:-2])
    k2          = (age_cool[2:] - age_cool[1:-1]) / (bp_rp[2:] - bp_rp[1:-1])
    k           = k1 + (bp_rp[1:-1] - bp_rp[:-2]) * (k1-k2) / (bp_rp[:-2]-bp_rp[2:])
    cool_rate   = np.concatenate(( np.array([1]), k , np.array([1]) ))
    
    
    # Get Parameters on HR Diagram
    grid_HR_to_mass, HR_to_mass         = interp_HR_to_para(bp_rp, G, mass_array, 
                                                            age, HR_grid, interp_type)
    grid_HR_to_logg, HR_to_logg         = interp_HR_to_para(bp_rp, G, logg, 
                                                            age, HR_grid, interp_type)
    grid_HR_to_age, HR_to_age           = interp_HR_to_para(bp_rp, G, age, 
                                                            age, HR_grid, interp_type)
    grid_HR_to_age_cool, HR_to_age_cool = interp_HR_to_para(bp_rp, G, age_cool, 
                                                            age, HR_grid, interp_type)
    grid_HR_to_logteff, HR_to_logteff   = interp_HR_to_para(bp_rp, G, logteff, 
                                                            age, HR_grid, interp_type)
    grid_HR_to_Mbol, HR_to_Mbol         = interp_HR_to_para(bp_rp, G, Mbol, 
                                                            age, HR_grid, interp_type)
#     row,col = grid_mass.shape
#     grid_mass_density                     = np.concatenate((np.zeros((row,1)),
#                                                             grid_mass[:,2:] - grid_mass[:,:-2],
#                                                             np.zeros((row,1)) ), axis=1)
    grid_HR_to_cool_rate, HR_to_cool_rate=interp_HR_to_para(bp_rp, G, cool_rate,
                                                            age, HR_grid, interp_type)
    # (mass, t_cool) --> bp-rp, G
    m_agecool_to_bprp                   = interp_xy_z_func(mass_array, age_cool,
                                                           bp_rp, interp_type )
    m_agecool_to_G                      = interp_xy_z_func(mass_array, age_cool,
                                                           G, interp_type )
    
    
    # Return a dictionary containing all the cooling track data points, 
    # interpolation functions and interpolation grids 
    return {'grid_logteff_logg_to_G_Mbol':grid_logteff_logg_to_G_Mbol,
            'logteff_logg_to_G_Mbol':logteff_logg_to_G_Mbol,
            'grid_logteff_logg_to_bp_rp':grid_logteff_logg_to_bp_rp,
            'logteff_logg_to_bp_rp':logteff_logg_to_bp_rp,
            'mass_array':mass_array, 'logg':logg, 'logteff':logteff,
            'age':age, 'age_cool':age_cool, 'cool_rate':cool_rate,
            'Mbol':Mbol, 'G':G, 'bp_rp':bp_rp,
            'grid_HR_to_mass':grid_HR_to_mass, 'HR_to_mass':HR_to_mass,
            'grid_HR_to_logg':grid_HR_to_logg, 'HR_to_logg':HR_to_logg,
            'grid_HR_to_age':grid_HR_to_age, 'HR_to_age':HR_to_age,
            'grid_HR_to_age_cool':grid_HR_to_age_cool, 'HR_to_age_cool':HR_to_age_cool,
            'grid_HR_to_logteff':grid_HR_to_logteff, 'HR_to_logteff':HR_to_logteff,
            'grid_HR_to_Mbol':grid_HR_to_Mbol, 'HR_to_Mbol':HR_to_Mbol,
            'grid_HR_to_cool_rate':grid_HR_to_cool_rate, 'HR_to_cool_rate':HR_to_cool_rate,
            'm_agecool_to_bprp':m_agecool_to_bprp, 
            'm_agecool_to_G':m_agecool_to_G}
