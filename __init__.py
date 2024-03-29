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

21 Jul 2019


Updates:

Oct 6, 2019:
We updated the pre-WD lifetime estimate.

Nov 1, 2019:
I updated the synthetic color table with much more pass bands (PanSTARRS, 
WISE, Spitzer, GALEX, etc.) and better atmosphere model (Blouin et al. 2018).
These updates directly come from the recent (Aug 2019) updates of the Montreal 
atmosphere models: http://www.astro.umontreal.ca/~bergeron/CoolingModels/.
Note that the filter names are slightly different from the old version, see 
the README document or the documentation in this script.

Jan 15, 2020:
I standardized the structure of this module and the installation process. 
The README.md document was also updated. 
In addition, since the Aug 2019 update of the Montreal atmosphere models
lacks information to log(g)~9.5, which makes inconvenience for analysing WDs
from Gaia, in my code, I duplicate the table for log(g)=9.0 as a reasonable 
guess for the photometric behaviour around log(g)=9.5. If the user is 
unconfortable with this extrapolation, he/she can import the old version:
`from WD_models import WD_models_old`. 

Feb 22, 2021:
I updated the synthetic color table with Gaia EDR3 passbands. These updates 
directly came from the recent (Jan 2021) updates of the Montreal atmosphere 
models: http://www.astro.umontreal.ca/~bergeron/CoolingModels/. Note that the
Gaia filter names are different (G2, bp2, G3, etc), see the README document
or the documentation in this script. Also note that now the default HR
coordinates are (bp3-rp3, M_G3).
In this updates, the Montreal group provides data for M=1.1 and M=1.3Msun, 
the latter corresponds to roughly log(g)= 9.2-9.3. I therefore used these 
data as a reasonable guess for the photometric behaviour around log(g)=9.5, 
similar to last version.
Last version is stored as 'WD_models_old.py'. To use it, please import:
`from WD_models import WD_models_2020`. 

"""

import os

import numpy as np
from astropy.table import Table, vstack
from scipy.interpolate import CloughTocher2DInterpolator, LinearNDInterpolator
from scipy.interpolate import griddata, interp1d

dirpath = os.path.dirname(__file__)

#-------------------------------------------------------------------------------
#
#   Define the functions that will be used for reading cooling tracks and 
#   interpolating the mappings 
#
#-------------------------------------------------------------------------------

# define the initial-final mass relation for calculating the total age for 
# some models
IFMR        = interp1d((0.19, 0.4, 0.50, 0.72, 0.87, 1.25, 1.4),
                       (0.23, 0.5, 0.95, 2.8, 3.65, 8.2, 10),
                       fill_value = 0, bounds_error=False)
def MS_age(m_WD):
    with np.errstate(divide='ignore', invalid='ignore'):
#        result = ( 10**9.38 * IFMR(m_WD)**-2.16 ) * (IFMR(m_WD) >= 2.3) + \
#            ( 10**10 * IFMR(m_WD)**-3.5 ) * (IFMR(m_WD) < 2.3)
# we update this pre-WD lifetime estimate on Oct 6, 2019.
        mi = IFMR(m_WD)
        if mi <= 2.11:
            life0 = 10**(13.37807 - 6.292517 * mi + 4.451837 * mi**2
                    - 1.773315 * mi**3 + 0.2944963 * mi**4)
        else: 
            life0 = 10**(10.75941 - 1.043523 * mi + 0.1366088 * mi**2
                    - 7.110290e-3 * mi**3)
    return life0
MS_age = np.vectorize(MS_age)

def interpolate_2d(x, y, z, method):
    if method == 'linear':
        interpolator = LinearNDInterpolator
    elif method == 'cubic':
        interpolator = CloughTocher2DInterpolator
    return interpolator((x,y), z, rescale=True)
    #return interp2d(x, y, z, kind=method)

def read_atm_table(atm_type):
    if atm_type == 'H':
        suffix = 'DA'
    if atm_type == 'He':
        suffix = 'DB'

    # read data from the Table_DA and Table_DB files
    atm_color_table = Table.read(dirpath+'/Montreal_atm_grid_2021/Table_'+suffix, format='ascii')
    # read the table for each mass
    # I assume the color information in this table is from the interpolation of the above table
    for mass in ['0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0','1.1','1.2','1.3']:
        atm_color_table_temp = Table.read(dirpath+'/Montreal_atm_grid_2021/Table_Mass_'+mass+'_'+suffix, format='ascii')
        atm_color_table_temp['M/Mo'] = float(mass)
        atm_color_table = vstack(( atm_color_table, atm_color_table_temp ))
    # add a table for ultra-massive WD with logg=9.5
    table_95 = Table.read(dirpath+'/Montreal_atm_grid_2021/Table_'+'Mass_1.3_'+suffix, format='ascii')
    table_95['logg'] = 9.5
    table_95['M/Mo'] = 1.366
    for column in [
        'Mbol','U','B','V','R','I','J','H','Ks','MY','MJ','MH','MK','W1','W2','W3','W4',
        'S3.6','S4.5','S5.8','S8.0','Su','Sg','Sr','Si','Sz','Pg','Pr','Pi','Pz','Py',
        'G2','G2_BP','G2_RP','G3','G3_BP','G3_RP','FUV','NUV'
    ]:
        table_95[column] += 1.108 - 0.652 # the effect of radius when converting M=1.3 to M=1.366
    atm_color_table = vstack(( atm_color_table, table_95 ))
    # add a column for logteff
    atm_color_table['logteff'] = np.log10(atm_color_table['Teff'])
    return atm_color_table
    
def interp_atm(atm_type, color, logteff_logg_grid=(3.5, 5.1, 0.01, 6.5, 9.6, 0.01), 
               interp_type_atm='linear'):
    """interpolate the mapping (logteff, logg) --> photometry
    
    This function interpolates the mapping (logteff, logg) --> color index or
    bolometric correction (BC) of a passband.
    
    Args:
        atm_type:           String. {'H', 'He'}
            See the "Synthetic color" section on http://www.astro.umontreal.ca/~bergeron/CoolingModels/
        color:              String. {'G3-Mbol', 'bp3-rp3', 'V-Mbol', 'u-g', 'B-z', etc.}
            The target color of the mapping. Any two bands between 
            Gaia EDR3: G3, bp3, rp3
            Gaia DR2: G2, bp2, rp2
            SDSS: Su, Sg, Sr, Si, Sz
            PanSTARRS: Pg, Pr, Pi, Pz, Py
            Johnson: U, B, V, R, I
            2MASS: J, H, Ks
            Mauna Kea Observatory (MKO): MY, MJ, MH, MK
            WISE: W1, W2, W3, W4
            Spitzer: S36, S45, S58, S80
            GALEX: FUV, NUV
            and the bolometric magnitude: Mbol.
            For bolometric correction (BC), 'Mbol' must be the passband after
            the minus sign '-'.
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
    atm_color_table = read_atm_table(atm_type)
    selected  = atm_color_table['Teff'] > 10**logteff_logg_grid[0]
    atm_color_table = atm_color_table[selected]
    
    grid_x, grid_y = np.mgrid[logteff_logg_grid[0]:logteff_logg_grid[1]:logteff_logg_grid[2],
                              logteff_logg_grid[3]:logteff_logg_grid[4]:logteff_logg_grid[5]]
    
    band_name_column_list = ['U','B','V','R','I','J','H','Ks','MY','MJ','MH','MK','W1','W2','W3','W4',
        'S3.6','S4.5','S5.8','S8.0','Su','Sg','Sr','Si','Sz','Pg','Pr','Pi','Pz','Py',
        'G2','G2_BP','G2_RP','G3','G3_BP','G3_RP','FUV','NUV']
    band_name_list = ['U','B','V','R','I','J','H','Ks','MY','MJ','MH','MK','W1','W2','W3','W4',
        'S36','S45','S58','S80','Su','Sg','Sr','Si','Sz','Pg','Pr','Pi','Pz','Py',
        'G2','bp2','rp2','G3','bp3','rp3','FUV','NUV']
    band_name_dict = {}
    for i,band_name in enumerate(band_name_list):
        band_name_dict[band_name] = band_name_column_list[i]
    
    # define the interpolation of mapping
    def interp(x, y, z, interp_type_atm='linear'):
        grid_z      = griddata(np.array((x, y)).T, z, (grid_x, grid_y),
                               method=interp_type_atm)
        z_func      = interpolate_2d(x, y, z, interp_type_atm)
        return grid_z, z_func
    
    division = color.find('-')
    if '-Mbol' in color:
        z = atm_color_table[band_name_dict[color[:division]]] -  atm_color_table['Mbol']
    else:
        z = atm_color_table[band_name_dict[color[:division]]] - atm_color_table[band_name_dict[color[division+1:]]]
    
    return interp(atm_color_table['logteff'], atm_color_table['logg'], z, interp_type_atm)


def read_cooling_tracks(low_mass_model, middle_mass_model, high_mass_model,
                        atm_type, logg_func=None, for_comparison=False):
    """ Read a set of cooling tracks
    
    This function reads the cooling models and stack together the data points
    of mass, logg, age, age_cool, logteff, and Mbol from different cooling
    tracks.
    
    Args:
        low_mass_model:     String. 
            Specifying the cooling model used for low-mass WDs (<~0.5Msun). 
        middle_mass_model:  String. 
            Specifying the cooling model used for middle-mass WDs (about 
            0.5~1.0Msun)
        high_mass_model:    String. 
            Specifying the cooling model used for high-mass WDs (>~1.0Msun).
        atm_type:           String. {'H', 'He'}
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
    if ('Renedo2010_' in middle_mass_model or 
        middle_mass_model == 'Camisassa2017' or 
        middle_mass_model == 'PG'
       ):
        if for_comparison == True:
            mass_separation_1 = 0.501
        else:
            mass_separation_1 = 0.45
    if 'BaSTI' in middle_mass_model:
        mass_separation_1 = 0.501
    
    if high_mass_model == 'Fontaine2001' or high_mass_model == 'Fontaine2001_thin':
        mass_separation_2 = 0.99
    if high_mass_model == 'ONe':
        mass_separation_2 = 1.09
    if high_mass_model == 'MESA':
        if for_comparison == True:
            mass_separation_2 = 1.01
        else:
            mass_separation_2 = 0.99
    if high_mass_model == 'BaSTI' or high_mass_model == 'BaSTInosep':
        mass_separation_2 = 0.99
    
    # initialize data points of cooling tracks
    logg        = np.zeros(0)
    age         = np.zeros(0) 
    age_cool    = np.zeros(0)
    logteff     = np.zeros(0)
    mass_array  = np.zeros(0)
    Mbol        = np.zeros(0)
    
    if atm_type == 'H':
        spec_suffix2 = 'DA'
    if atm_type == 'He':
        spec_suffix2 = 'DB'
    
    # read data from cooling models
    # Bedard et al. 2020
    for mass in ['020','025','030','035','040','045','050','055','060',
             '065','070','075','080','085','090','095','100',
             '105','110','115','120','125','130']:
        if int(mass)/100 < mass_separation_1 and atm_type == 'H':
            if low_mass_model == 'Bedard2020':
                spec_suffix = '_thick'
            elif low_mass_model == 'Bedard2020_thin':
                spec_suffix = '_thin'
            else: continue
        elif (int(mass)/100 > mass_separation_1 and 
              int(mass)/100 < mass_separation_2 and atm_type == 'H'):
            if middle_mass_model == 'Bedard2020':
                spec_suffix = '_thick'
            elif middle_mass_model == 'Bedard2020_thin':
                spec_suffix = '_thin'
            else: continue
        elif int(mass)/100 > mass_separation_2 and atm_type == 'H':
            if high_mass_model == 'Bedard2020':
                spec_suffix = '_thick'
            elif high_mass_model == 'Bedard2020_thin':
                spec_suffix = '_thin'
            else: continue
        else: continue
        f       = open(dirpath+'/cooling_models/Bedard_AllSequences/seq_'
                        + mass + spec_suffix + '.txt')
        title_offset = 76*5-4
        text    = f.read()[title_offset:]
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
                                  MS_age(int(mass)/100) )
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
    
    # Montreal He-atmosphere model (2021 verion)
    for mass in ['0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0','1.1','1.2','1.3']:
        if ((float(mass) < mass_separation_1 and
             low_mass_model == 'Bedard2020') or
            (float(mass) > mass_separation_1 and
             float(mass) < mass_separation_2 and
             middle_mass_model == 'Bedard2020') or
            (float(mass) > mass_separation_2 and
             high_mass_model == 'Bedard2020')
           ) and atm_type == 'He':
            Cool = Table.read(dirpath+'/Montreal_atm_grid_2021/Table_Mass_' + mass +
                                    '_'+spec_suffix2, format='ascii')
            Cool = Cool[::1]
            mass_array  = np.concatenate(( mass_array, np.ones(len(Cool))*float(mass) ))
            logg        = np.concatenate(( logg, Cool['logg'] ))
            age         = np.concatenate(( age, Cool['Age'] +
                                                MS_age(float(mass)) ))
            age_cool    = np.concatenate(( age_cool, Cool['Age'] ))
            logteff     = np.concatenate(( logteff, np.log10(Cool['Teff']) ))
            Mbol        = np.concatenate(( Mbol, Cool['Mbol'] ))

    # Fontaine et al. 2001
    for mass in ['020','030','040','050','060','070','080','090','095','100',
                 '105','110','115','120','125','130']:
        if int(mass)/100 < mass_separation_1 and atm_type == 'H':
            if low_mass_model == 'Fontaine2001':
                spec_suffix = '0204'
            elif low_mass_model == 'Fontaine2001_thin':
                spec_suffix = '0210'
            else: continue
        elif (int(mass)/100 > mass_separation_1 and 
              int(mass)/100 < mass_separation_2 and atm_type == 'H'):
            if middle_mass_model == 'Fontaine2001':
                spec_suffix = '0204'
            elif middle_mass_model == 'Fontaine2001_thin':
                spec_suffix = '0210'
            else: continue
        elif int(mass)/100 > mass_separation_2 and atm_type == 'H':
            if high_mass_model == 'Fontaine2001':
                spec_suffix = '0204'
            elif high_mass_model == 'Fontaine2001_thin':
                spec_suffix = '0210'
            else: continue
        else: continue
        f       = open(dirpath+'/cooling_models/Fontaine_AllSequences/CO_'
                        + mass + spec_suffix)
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
                                  MS_age(int(mass)/100) )
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
    
    # Montreal He-atmosphere model
    for mass in ['0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0','1.2']:
        if ((float(mass) < mass_separation_1 and
             low_mass_model == 'Fontaine2001') or
            (float(mass) > mass_separation_1 and
             float(mass) < mass_separation_2 and
             middle_mass_model == 'Fontaine2001') or
            (float(mass) > mass_separation_2 and
             high_mass_model == 'Fontaine2001')
           ) and atm_type == 'He':
            Cool = Table.read(dirpath+'/Montreal_atm_grid_2019/Table_Mass_' + mass +
                                    '_'+spec_suffix2, format='ascii')
            Cool = Cool[::1]
            mass_array  = np.concatenate(( mass_array, np.ones(len(Cool))*float(mass) ))
            logg        = np.concatenate(( logg, Cool['logg'] ))
            age         = np.concatenate(( age, Cool['Age'] +
                                                MS_age(float(mass)) ))
            age_cool    = np.concatenate(( age_cool, Cool['Age'] ))
            logteff     = np.concatenate(( logteff, np.log10(Cool['Teff']) ))
            Mbol        = np.concatenate(( Mbol, Cool['Mbol'] ))
    
    # define a smoothing function for future extension. Now it just returns the
    # input x vector.
    def smooth(x,window_len=5,window='hanning'):
        w=eval('np.'+window+'(window_len)')
        y=np.convolve(w/w.sum(),x,mode='same')
        return x
    
    # CO, DA (Renedo et al. 2010)
    if 'Renedo2010_' in middle_mass_model and atm_type == 'H':
        if '_001' in middle_mass_model:
            Renedo_masslist = ['0524','0570','0593','0609','0632','0659',
                                '0705','0767','0837','0877','0934']
            metallicity = '001'
        if '_0001' in middle_mass_model:
            Renedo_masslist = ['0505','0553','0593','0627','0660','0692',
                                '0863']
            metallicity = '0001'
        for mass in Renedo_masslist:
            Cool = Table.read(dirpath+'/cooling_models/Renedo_2010_DA_CO/wdtracks_z' + 
                              metallicity + '/wd' + mass + '_z' + metallicity +
                              '.trk', format='ascii') 
            Cool = Cool[::5] #[(Cool['log(TEFF)'] > logteff_min) * (Cool['log(TEFF)'] < logteff_max)]
            mass_array  = np.concatenate(( mass_array, np.ones(len(Cool))*int(mass)/1000 ))
            logg        = np.concatenate(( logg, Cool['Log(grav)'] ))
            age         = np.concatenate(( age, Cool['age/Myr'] * 1e6 +
                                                MS_age(int(mass)/1000) ))
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
    if middle_mass_model == 'Camisassa2017' and atm_type == 'He':
        for mass in ['051','054','058','066','074','087','100']:
            if int(mass)/100 < mass_separation_2:
                Cool = Table.read(dirpath+'/cooling_models/Camisassa_2017_DB_CO/Z002/' + mass +
                                  'DB.trk', format='ascii') 
                dn = 1
                if int(mass)/100 > 0.95:
                    dn = 50
                Cool = Cool[::dn] # [(Cool['LOG(TEFF)'] > logteff_min) * (Cool['LOG(TEFF)'] < logteff_max)]
                #Cool.sort('Log(edad/Myr)')
                mass_array  = np.concatenate(( mass_array, np.ones(len(Cool)) * int(mass)/100 ))
                logg        = np.concatenate(( logg, Cool['Log(grav)'] ))
                age         = np.concatenate(( age, 10**Cool['Log(edad/Myr)'] * 1e6 ))
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
            if middle_mass_model == 'BaSTI':
                sep = 'sep'
            elif middle_mass_model == 'BaSTI_nosep':
                sep = 'nosep'
            else: continue
        elif int(mass)/100 > mass_separation_2:
            if high_mass_model == 'BaSTI':
                sep = 'sep'
            elif high_mass_model == 'BaSTI_nosep':
                sep = 'nosep'
            else: continue
        else: continue
        Cool = Table.read(dirpath+'/cooling_models/BaSTI/COOL' + mass + 'BaSTIfinale' + 
               spec_suffix2 + sep +'.sdss', format='ascii')
        dn = 1
        if int(mass)/100 > 1.05:
            dn = 5
        Cool = Cool[::dn] # [(Cool['log(Teff)'] > logteff_min) * (Cool['log(Teff)'] < logteff_max)]
        #Cool.sort('Log(edad/Myr)')
        Cool['Log(grav)'] = logg_func(Cool['log(Teff)'], np.ones(len(Cool)) * int(mass)/100)
        mass_array  = np.concatenate(( mass_array, np.ones(len(Cool)) * int(mass)/100 ))
        logg        = np.concatenate(( logg, Cool['Log(grav)'] ))
        age         = np.concatenate(( age, 10**Cool['log(t)'] +
                                            MS_age(int(mass)/100) ))
        age_cool    = np.concatenate(( age_cool, 10**Cool['log(t)'] ))
        logteff     = np.concatenate(( logteff, Cool['log(Teff)'] ))
        Mbol        = np.concatenate(( Mbol, 4.75 - 2.5 * Cool['log(L/Lo)'] ))
        del Cool

    # Ultra-massive ONe model (Camisassa et al. 2019)
    if high_mass_model == 'ONe':
        for mass in ['110','116','122','129']:
            Cool = Table.read(dirpath+'/cooling_models/ONeWDs/' + mass + '_' + spec_suffix2 +
                              '.trk', format='ascii') 
            Cool = Cool[::10] # (Cool['LOG(TEFF)'] > logteff_min) * (Cool['LOG(TEFF)'] < logteff_max)
            #Cool.sort('Log(edad/Myr)')
            mass_array  = np.concatenate(( mass_array, np.ones(len(Cool)) * int(mass)/100 ))
            logg        = np.concatenate(( logg, Cool['Log(grav)'] ))
            age         = np.concatenate(( age, (10**Cool['Log(edad/Myr)'] -
                                                 10**Cool['Log(edad/Myr)'][0]) * 1e6 + \
                                                 MS_age(int(mass)/100) ))
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
#        for mass in ['135','138']:
#            Cool = Table.read('models/t' + mass + '.trk', format='ascii') 
#            Cool = Cool[::1] # (Cool['LOG(TEFF)'] > logteff_min) * (Cool['LOG(TEFF)'] < logteff_max)
#            #Cool.sort('Log(edad/Myr)')
#            mass_array  = np.concatenate(( mass_array, np.ones(len(Cool)) * int(mass)/100 ))
#            logg        = np.concatenate(( logg, np.ones(len(Cool)) * 9.4999 ))
#            age         = np.concatenate(( age, Cool['col7']* 1e6 + MS_age(int(mass)/100) ))
#            age_cool    = np.concatenate(( age_cool, Cool['col7'] * 1e6 ))
#            logteff     = np.concatenate(( logteff, Cool['col2'] ))
#            Mbol        = np.concatenate(( Mbol, 4.75 - 2.5 * Cool['col1'] ))
#            del Cool
    
    # massive MESA model (Lauffer et al. 2019)
    if high_mass_model == 'MESA': 
        if atm_type == 'He':
            mesa_masslist = ['1.0124','1.019','1.0241','1.0358','1.0645','1.0877',
                             '1.1102','1.1254','1.1313','1.1322','1.1466','1.151',
                             '1.2163','1.22','1.2671','1.3075']
#                              ['1.0124','1.0645',
#                              '1.1102','1.151',
#                              '1.2163','1.2671','1.3075']
        if atm_type == 'H':
            mesa_masslist = ['1.0124','1.019','1.0241','1.0358','1.0645','1.0877',
                             '1.1102','1.125','1.1309','1.1322','1.1466','1.151',
                             '1.2163','1.22','1.2671','1.3075']
#                              ['1.0124','1.0645',
#                              '1.1102','1.151',
#                              '1.2163','1.2671','1.3075']
        for mass in mesa_masslist:
            Cool = Table.read(dirpath+'/cooling_models/MESA_model/' + atm_type + '_atm-M' +
                              mass + '.dat',
                              format='csv', header_start=1, data_start=2)
            dn = 70
            if float(mass) > 1.2:
                dn = 120
            if float(mass) < 1.05 or atm_type == 'He':
                dn = 10
            dn = len(Cool)//40
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
    with np.errstate(divide='ignore', invalid='ignore'):
        selected = ~np.isnan(color + Mag + WD_para) * \
                   (Mag > HR_grid[3]) * (Mag < HR_grid[4]) * \
                   (color > HR_grid[0]) * (color < HR_grid[1])
    
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
    x and y values, but does not calculate the grid values as interp_HR_to_para
    and interp_xy_z do.
    
    Args:
        x:              1d-array. The x in the mapping (x, y) --> z
        y:              1d-array. The y in the mapping (x, y) --> z
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


def load_model(low_mass_model, middle_mass_model, high_mass_model, atm_type,
               HR_bands=('bp3-rp3', 'G3'),
               HR_grid=(-0.6, 1.5, 0.002, 8, 18, 0.01),
               logteff_logg_grid=(3.5, 5.1, 0.01, 6.5, 9.6, 0.01),
               interp_type_atm='linear', interp_type='linear',
               for_comparison=False):
    """ Load a set of cooling tracks and interpolate the HR diagram mapping
    
    This function reads a set of cooling tracks assigned by the user and returns
    many useful grid-values for plotting the contour of WD parameters on the
    H--R diagram and functions for mapping between photometry and WD parameters.
    
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
            'Bedard2020' or 'be'            the updated thick-H- or He-atmosphere CO WD model in 
                                            http://www.astro.umontreal.ca/~bergeron/CoolingModels/
            'Bedard2020_thin' or 'bet'      the updated thin-H CO WD model in 
                                            http://www.astro.umontreal.ca/~bergeron/CoolingModels/
            'Fontaine2001' or 'f'           the old thick-H- or He-atmosphere CO WD model in 
                                            http://www.astro.umontreal.ca/~bergeron/CoolingModels/
            'Fontaine2001_thin' or 'ft'     the old thin-H CO WD model in 
                                            http://www.astro.umontreal.ca/~bergeron/CoolingModels/
        middle_mass_model:  String. 
                            Specifying the cooling model used for middle-mass
                            WDs (about 0.5~1.0Msun). Its value should be one of
                            the following:
            ''                              no middle-mass model will be read
            'Bedard2020' or 'be'            the updated thick-H- or He-atmosphere CO WD model in 
                                            http://www.astro.umontreal.ca/~bergeron/CoolingModels/
            'Bedard2020_thin' or 'bet'      the updated thin-H CO WD model in 
                                            http://www.astro.umontreal.ca/~bergeron/CoolingModels/
            'Fontaine2001' or 'f'           the old thick-H- or He-atmosphere CO WD model in 
                                            http://www.astro.umontreal.ca/~bergeron/CoolingModels/
            'Fontaine2001_thin' or 'ft'     the old thin-H CO WD model in 
                                            http://www.astro.umontreal.ca/~bergeron/CoolingModels/
            'Renedo2010_001' or 'r001'     Z=0.01, only for DA, http://evolgroup.fcaglp.unlp.edu.ar/TRACKS/tracks_cocore.html
            'Renedo2010_0001' or 'r0001'   Z=0.001, only for DA, http://evolgroup.fcaglp.unlp.edu.ar/TRACKS/tracks_cocore.html
            'Camisassa2017' or 'c'          only for DB, http://evolgroup.fcaglp.unlp.edu.ar/TRACKS/tracks_DODB.html
            'BaSTI' or 'b'                  with phase separation, Salaris et al. 2010, http://basti.oa-teramo.inaf.it
            'BaSTI_nosep' or 'bn'           no phase separation, Salaris et al. 2010, http://basti.oa-teramo.inaf.it
        high_mass_model:    String. 
                            Specifying the cooling model used for high-mass WDs
                            (>~1.0Msun). Should be one of the following: 
            ''                              no high-mass model will be read
            'Bedard2020' or 'be'            the updated thick-H- or He-atmosphere CO WD model in 
                                            http://www.astro.umontreal.ca/~bergeron/CoolingModels/
            'Bedard2020_thin' or 'bet'      the updated thin-H CO WD model in 
                                            http://www.astro.umontreal.ca/~bergeron/CoolingModels/
            'Fontaine2001' or 'f'           the old thick-H- or He-atmosphere CO WD model in 
                                            http://www.astro.umontreal.ca/~bergeron/CoolingModels/
            'Fontaine2001_thin' or 'ft'     the old thin-H CO WD model in 
                                            http://www.astro.umontreal.ca/~bergeron/CoolingModels/
            'ONe' or 'o'                    Camisassa et al. 2019, http://evolgroup.fcaglp.unlp.edu.ar/TRACKS/ultramassive.html
            'MESA' or 'm'                   Lauffer et al. 2018
            'BaSTI' or 'b'                  with phase separation, Salaris et al. 2010, http://basti.oa-teramo.inaf.it
            'BaSTI_nosep' or 'bn'           no phase separation, Salaris et al. 2010, http://basti.oa-teramo.inaf.it
        atm_type:           String. 
                            Specifying the atmosphere composition. Its value 
                            should be one of the following:
            'H'                             pure-hydrogen atmosphere
            'He'                            pure-helium atmosphere
        HR_bands:          (String, String). *Optional*
            The passbands for the color and absolute magnitude on the H--R
            diagram. It can be any combination from the following bands:
                G3, bp3, rp3 (Gaia EDR3); G2, bp2, rp2 (Gaia DR2); 
                Su, Sg, Sr, Si, Sz (SDSS); Pg, Pr, Pi, Pz, Py (PanSTARRS); 
                U, B, V, R, I (Johnson); J, H, K (2MASS);
                MY, MJ, MH, MK (Mauna Kea Observatory (MKO));
                W1, W2, W3, W4 (WISE); S36, S45, S58, S80 (SPITZER);
                FUV, NUV (GALEX)
            Color should be in the format of:
                'bp3-rp3', 'G3-bp3', 'U-B', 'u-g', 'G3-i', etc. 
            Absolute magnitude should be in the format of:
                'G3', 'bp3', 'U', 'u', etc. 
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
    if low_mass_model == 'be':
        low_mass_model = 'Bedard2020'
    if low_mass_model == 'bet':
        low_mass_model = 'Bedard2020_thin'
    if low_mass_model == 'f':
        low_mass_model = 'Fontaine2001'
    if low_mass_model == 'ft':
        low_mass_model = 'Fontaine2001_thin'
    
    if middle_mass_model == 'r001':
        middle_mass_model = 'Renedo2010_001'
    if middle_mass_model == 'r0001':
        middle_mass_model = 'Renedo2010_0001'
    if middle_mass_model == 'be':
        middle_mass_model = 'Bedard2020'
    if middle_mass_model == 'bet':
        middle_mass_model = 'Bedard2020_thin'
    if middle_mass_model == 'f':
        middle_mass_model = 'Fontaine2001'
    if middle_mass_model == 'ft':
        middle_mass_model = 'Fontaine2001_thin'
    if middle_mass_model == 'c':
        middle_mass_model = 'Camisassa2017'
    if middle_mass_model == 'b':
        middle_mass_model = 'BaSTI'
    if middle_mass_model == 'bn':
        middle_mass_model = 'BaSTI_nosep'
    
    if high_mass_model == 'be':
        high_mass_model = 'Bedard2020'
    if high_mass_model == 'bet':
        high_mass_model = 'Bedard2020_thin'
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
    
    model_names = [
        '', 'Bedard2020', 'Bedard2020_thin',
        'Fontaine2001', 'Fontaine2001_thin', 
        'Renedo2010_001', 'Renedo2010_0001', 'Camisassa2017', 
        'BaSTI', 'BaSTI_nosep',
        'MESA', 'ONe',]
    if (low_mass_model not in model_names or 
        middle_mass_model not in model_names or
        high_mass_model not in model_names):
        print('please check the model names.')
    if atm_type not in ['H', 'He']:
        print('please enter either \'H\' or \'He\' for atm_type.')
    
    # make atmosphere grid and mapping: logteff, logg --> bp3-rp3,  G3-Mbol
    grid_logteff_logg_to_color, logteff_logg_to_color = interp_atm(
        atm_type, HR_bands[0], 
        logteff_logg_grid=logteff_logg_grid,
        interp_type_atm=interp_type_atm)
    grid_logteff_logg_to_BC, logteff_logg_to_BC = interp_atm(
        atm_type, HR_bands[1] + '-Mbol', 
        logteff_logg_grid=logteff_logg_grid,
        interp_type_atm=interp_type_atm)


    # get for logg_func BaSTI models
    if 'BaSTI' in middle_mass_model or 'BaSTI' in high_mass_model:
        mass_array_Bedard2020, logg_Bedard2020, _, _, logteff_Bedard2020, _\
                    = read_cooling_tracks('Bedard2020',
                                          'Bedard2020',
                                          'Bedard2020',
                                          atm_type)
        logg_func   = interp_xy_z_func(x=logteff_Bedard2020,
                                       y=mass_array_Bedard2020,
                                       z=logg_Bedard2020)
    else: 
        logg_func   = None
    
    # Open Evolutionary Tracks
    mass_array, logg, age, age_cool, logteff, Mbol \
                    = read_cooling_tracks(low_mass_model,
                                          middle_mass_model,
                                          high_mass_model,
                                          atm_type, logg_func, for_comparison)

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
    # (mass, t_cool) --> bp3-rp3, G3
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


def read_crystallization_fraction(
    HR_bands=('bp3-rp3', 'G3'),
    HR_grid=(-0.6, 1.5, 0.002, 8, 18, 0.01),
    logteff_logg_grid=(3.5, 5.1, 0.01, 6.5, 9.6, 0.01),
    interp_type_atm='linear', interp_type='linear',
    for_comparison=False):
    
    atm_type = 'H'
    # make atmosphere grid and mapping: logteff, logg --> bp3-rp3,  G3-Mbol
    grid_logteff_logg_to_color, logteff_logg_to_color = interp_atm(
        atm_type, HR_bands[0], 
        logteff_logg_grid=logteff_logg_grid,
        interp_type_atm=interp_type_atm)
    grid_logteff_logg_to_BC, logteff_logg_to_BC = interp_atm(
        atm_type, HR_bands[1] + '-Mbol', 
        logteff_logg_grid=logteff_logg_grid,
        interp_type_atm=interp_type_atm)

    logg        = np.zeros(0)
    logteff     = np.zeros(0)
    Mbol        = np.zeros(0)
    X           = np.zeros(0)
    for mass in ['020','030','040','050','060','070','080','090','095','100',
                 '105','110','115','120','125','130']:
        #f       = open('models/Fontaine_AllSequences/CO_' + mass + '0204')
        f       = open(dirpath+'/cooling_models/Fontaine_AllSequences/C_' + mass + '0204')
        text    = f.read()
        example = ('      1    57674.0025    8.36722799  7.160654E+08 '
                    ' 4.000000E+05  4.042436E+33\n'
                    '        7.959696E+00  2.425570E+01  7.231926E+00 '
                    ' 0.0000000000  0.000000E+00\n'
                    '        6.019629E+34 -4.010597E+00 -1.991404E+00 '
                    '-3.055254E-01 -3.055254E-01'
                  )
        logg_temp       = []
        logteff_temp    = []
        Mbol_temp       = []
        X_temp          = []
        l_line          = len(example)
        #for line in range(len(text)//l_line):
        for line in range(len(text)//l_line):
            logteff_temp.append(  np.log10(float(text[line*l_line+9:
                                                      line*l_line+21])) )
            logg_temp.append(     float(text[line*l_line+22:line*l_line+35]) )
            Mbol_temp.append(     4.75 - 
                                  2.5 * np.log10(float(text[line*l_line+64:
                                                            line*l_line+76]
                                                      ) / 3.828e33) )
            X_temp.append( float(text[line*l_line+79+48:line*l_line+79+60]) )
        logg        = np.concatenate(( logg, logg_temp ))
        logteff     = np.concatenate(( logteff, logteff_temp ))
        Mbol        = np.concatenate(( Mbol, Mbol_temp ))
        X           = np.concatenate(( X, X_temp ))
        f.close()
            
    # Get Colour/Magnitude for Evolution Tracks
    Mag         = logteff_logg_to_BC(logteff, logg) + Mbol
    color       = logteff_logg_to_color(logteff, logg)
    
    # Get Parameters on HR Diagram
    grid_HR_to_X, HR_to_X         = interp_HR_to_para(color, Mag, X, 
                                                            HR_grid, interp_type)
    return {'grid_HR_to_X':grid_HR_to_X, 'HR_to_X':HR_to_X}

