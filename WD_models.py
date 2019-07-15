'''
This package will read different cooling models and interpolate the conversion mappings
between HR diagram, Teff, Mbol, etc. The mappings are stored in dictionaries for each model.
See the main function and the lines after its definition.
This package also contains the functions to read a single cooling track.
'''


import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, vstack, hstack
from scipy.interpolate import interp1d, interp2d, CloughTocher2DInterpolator, griddata, LinearNDInterpolator


#----------------------------------------------------------------------------------------------------   
#
#   Define the functions that will be used for reading cooling tracks and interpolating the mappings 
#
#----------------------------------------------------------------------------------------------------


def interpolate_2d(x, y, z, method):
    if method == 'linear':
        interpolator = LinearNDInterpolator
    elif method == 'cubic':
        interpolator = CloughTocher2DInterpolator
    return interpolator((x,y), z, rescale=True)
  

def interp_atm(spec_type, color, T_logg_grid=(3.5, 5.1, 0.01, 6.5, 9.6, 0.01), interp_type_atm='linear'):
    '''
    This function generates from the atmosphere model the mapping (logteff, logg) --> G, BP-RP, or G-Mbol 
    
    Arguments:
    spec_type:    string. 'DA_thick' or 'DA_thin' or 'DB'. 
                  See http://www.astro.umontreal.ca/~bergeron/CoolingModels/
    color:        string. 'G', 'BP-RP', or 'G-Mbol'. This is the target photometry of the mapping. 
    T_logg_grid:  in the form (xmin, xmax, dx, ymin, ymax, dy), corresponding to the grid of logTeff and logg.
    '''
    logteff     = np.zeros(0)
    logg        = np.zeros(0)
    age         = np.zeros(0)
    mass_array  = np.zeros(0)
    G           = np.zeros(0)
    bp_rp       = np.zeros(0)
    Mbol        = np.zeros(0)
    
    # read the table for each mass
    for mass in ['0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0','1.2']:
        Atm_color = Table.read('models/Montreal_atm_grid/Table_Mass_'+mass+'_'+spec_type,
                               format='ascii')
        selected  = Atm_color['Teff'] > 3500
        Atm_color = Atm_color[selected]
        
        # read columns
        bp_rp   = np.concatenate(( bp_rp, Atm_color['G_BP/R'] - Atm_color['G_RP/R'] ))
        G       = np.concatenate(( G, Atm_color['G/R'] ))
        Mbol    = np.concatenate(( Mbol, Atm_color['Mbol'] ))
        logteff = np.concatenate(( logteff, np.log10(Atm_color['Teff']) ))
        logg    = np.concatenate(( logg, Atm_color['logg'] ))
        age     = np.concatenate(( age, Atm_color['Age'] ))
        
    # read the table for all logg
    Atm_color = Table.read('models/Montreal_atm_grid/Table_'+spec_type, format='ascii')
    selected  = Atm_color['Teff'] > 3500
    Atm_color = Atm_color[selected]
    
    # read columns
    bp_rp   = np.concatenate(( bp_rp, Atm_color['G_BP/R'] - Atm_color['G_RP/R'] ))
    G       = np.concatenate(( G, Atm_color['G/R'] ))
    Mbol    = np.concatenate(( Mbol, Atm_color['Mbol'] ))
    logteff = np.concatenate(( logteff,np.log10(Atm_color['Teff']) ))
    logg    = np.concatenate(( logg, Atm_color['logg'] ))
    age     = np.concatenate(( age, Atm_color['Age'] ))        
    
    grid_x, grid_y = np.mgrid[T_logg_grid[0]:T_logg_grid[1]:T_logg_grid[2],
                              T_logg_grid[3]:T_logg_grid[4]:T_logg_grid[5]]
    
    # define the interpolation of mapping
    def interp(x,y,z):
        grid_z      = griddata(np.array((x,y)).T, z, (grid_x, grid_y), method=interp_type_atm)
        grid_z_func = interpolate_2d(x, y, z, interp_type_atm)
        return grid_z, grid_z_func
    
    if color == 'G':
        return interp(logteff, logg, G)
    if color == 'bp_rp':
        return interp(logteff, logg, bp_rp)
    if color == 'G_Mbol':
        return interp(logteff, logg, G-Mbol)


def open_evolution_tracks(normal_mass_model, high_mass_model, spec_type, logg_func=None, 
                          for_comparison=False):
    '''
    Read the cooling models and store the following information of different cooling
    tracks together in one numpy array: mass, logg, age, age_cool, logteff, Mbol
    
    Arguments:
    normal_mass_model:  string. One of the following: 
                        'Fontaine2001'    (http://www.astro.umontreal.ca/~bergeron/CoolingModels/),
                        'Althaus2010_001', 'Althaus2010_0001'
                                          (Z=0.01 and Z=0.001, only for DA, 
                                           http://evolgroup.fcaglp.unlp.edu.ar/TRACKS/tracks_cocore.html),
                        'Camisassa2017'   (only for DB, 
                                           http://evolgroup.fcaglp.unlp.edu.ar/TRACKS/tracks_DODB.html),
                        'BaSTI', 'BaSTI_2' (Z=0.02), 'BaSTI_4' (Z=0.04) 
                                          (Salaris et al. 2010, 
                                           http://basti.oa-teramo.inaf.it),
                        'PG'              (only for DB).
    high_mass_model:    string. One of the following: 
                        'Fontaine2001' (http://www.astro.umontreal.ca/~bergeron/CoolingModels/),
                        'ONe' (Camisassa et al. 2019, 
                               http://evolgroup.fcaglp.unlp.edu.ar/TRACKS/ultramassive.html),
                        'MESA' (Lauffer et al. 2019),
                        'BaSTI' (Salaris et al. 2010, http://basti.oa-teramo.inaf.it)
    spec_type:          string. One of the following:
                        'DA_thick', 'DA_thin', 'DB'
    logg_func:          a function for (logteff, mass) --> logg. It is necessary only for BaSTI models,
                        because the BaSTI models do not directly provide log g information.
    for_comparison:     Bool. If true, more cooling tracks from different models will be used. E.g., 
                        the Fontaine2001 model has m_WD = [..., 0.95, 1.00, ...], and the MESA model has
                        m_WD = [1.0124, 1.019, ...]. If true, the Fontaine2001 1.00 cooling track will be
                        used; if false, it will not be used because it is too close to the MESA 1.0124 track.
    '''
    logg        = np.zeros(0)
    age         = np.zeros(0) 
    age_cool    = np.zeros(0)
    logteff     = np.zeros(0)
    mass_array  = np.zeros(0)
    Mbol        = np.zeros(0)

    IFMR        = interp1d((0.19, 0.3, 0.4, 0.50, 0.55, 0.65, 0.75, 0.85, 1.0, 1.25, 1.35),
                           (0.3, 0.5, 0.7, 0.95, 1, 2, 3, 3.5, 5, 8, 9),
                           fill_value = 0, bounds_error=False) # mass_WD, mass_ini
    t_index     = -3

    # determine which cooling tracks in a model to read
    mass_separation_1 = 1.4 # when using only Fontaine2001 model
    mass_separation_2 = 1.4 # when using only Fontaine2001 model
    if 'Althaus2010' in normal_mass_model or normal_mass_model == 'Camisassa2017' or \
       normal_mass_model == 'PG':
        if for_comparison == True:
            mass_seperation_1 = 0.501
        else:
            mass_seperation_1 = 0.45
    if 'BaSTI' in normal_mass_model:
        mass_seperation_1 = 0.501
    
    if high_mass_model == 'Fontaine2001':
        mass_separation_2 = 1.4
    if high_mass_model == 'ONe':
        mass_separation_2 = 1.09
    if high_mass_model == 'MESA':
        if for_comparison == True:
            mass_separation_2 = 1.01
        else:
            mass_separation_2 = 0.99
    if high_mass_model == 'BaSTI':
        mass_separation_2 = 0.99
    
    # define atmosphere
    if spec_type == 'DB':
        spec_suffix = '0210'; spec_suffix2 = 'DB'; spec_suffix3 = 'He'
    else:
        spec_suffix = '0204'; spec_suffix2 = 'DA'; spec_suffix3 = 'H'
    
    # read cooling tracks
    # Fontaine et al. 2001
    for mass in ['020','030','040','050','060','070','080','090','095','100','105',
                 '110','115','120','125','130']:
        if int(mass)/100 < np.min((mass_separation_1, mass_separation_2)):
            f       = open('models/Fontaine_AllSequences/CO_' + mass + spec_suffix)
            text    = f.read()
            example = "      1    57674.0025    8.36722799  7.160654E+08  4.000000E+05  4.042436E+33\n"+\
                      "        7.959696E+00  2.425570E+01  7.231926E+00  0.0000000000  0.000000E+00\n"+\
                      "        6.019629E+34 -4.010597E+00 -1.991404E+00 -3.055254E-01 -3.055254E-01"          
            logg_temp       = []
            age_temp        = []
            age_cool_temp   = []
            logteff_temp    = []
            Mbol_temp       = []
            l_line          = len(example)
            for line in range(len(text)//l_line):
                logteff_temp.append(  np.log10(float(text[line*l_line+9:line*l_line+21])) )
                logg_temp.append(     float(text[line*l_line+22:line*l_line+35]) )
                age_temp.append(      float(text[line*l_line+48:line*l_line+63]) + \
                                      (IFMR(int(mass)/100))**(t_index) * 1e10 )
                age_cool_temp.append( float(text[line*l_line+48:line*l_line+63]) )
                Mbol_temp.append(     4.75 - 2.5 * \
                                             np.log10(float(text[line*l_line+64:line*l_line+76]) / \
                                                      3.828e33) )
            mass_array  = np.concatenate(( mass_array, np.ones(len(logg_temp)) * int(mass)/100 ))
            logg        = np.concatenate(( logg, logg_temp ))
            age         = np.concatenate(( age, age_temp ))
            age_cool    = np.concatenate(( age_cool, age_cool_temp ))
            logteff     = np.concatenate(( logteff, logteff_temp ))
            Mbol        = np.concatenate(( Mbol, Mbol_temp ))
    
    # define a smoothing function for future extension. Now it just returns the input x vector.
    def smooth(x,window_len=5,window='hanning'):
        w=eval('np.'+window+'(window_len)')
        y=np.convolve(w/w.sum(),x,mode='same')
        return x
    
    # CO, DA (Althaus et al. 2010)
    if 'Althaus2010_' in normal_mass_model and 'DA' in spec_type:
        if '_001' in normal_mass_model:
            Althaus_masslist = ['0524','0570','0593','0609','0632','0659','0705','0767',
                                '0837','0877','0934']
            metallicity = '001'
        if '_0001' in normal_mass_model:
            Althaus_masslist = ['0505','0553','0593','0627','0660','0692','0863']
            metallicity = '0001'
        for mass in Althaus_masslist:
            Cool = Table.read('models/Althaus_2010_DA_CO/wdtracks_z' + metallicity + '/wd' + mass + \
                              '_z' + metallicity + '.trk', format='ascii') 
            Cool = Cool[(Cool['log(TEFF)'] > tmin) * (Cool['log(TEFF)'] < tmax)][::1]
            mass_array  = np.concatenate(( mass_array, np.ones(len(Cool)) * int(mass) / 1000 ))
            logg        = np.concatenate(( logg, smooth(np.array(Cool['Log(grav)'])) ))
            age         = np.concatenate(( age, Cool['age/Myr)'] * 1e6 + \
                                                (IFMR(int(mass)/1000))**(t_index) * 1e10 ))
            age_cool    = np.concatenate(( age_cool, Cool['age/Myr)'] * 1e6 ))
            logteff     = np.concatenate(( logteff, smooth(np.array(Cool['log(TEFF)'])) ))
            Mbol        = np.concatenate(( Mbol, 4.75 - 2.5 * smooth(np.array(Cool['log(L)'])) ))
            # additional
            logT_c      = Cool['logT_C']
            logrho_c    = Cool['logRo_c']
            mass_accurate= Cool['Mass']
            logL_nu     = Cool['Log(Lnu)']
            logMH       = Cool['LogMHtot']
            logr        = np.log10(Cool['R/R_sun'])
    
    # CO, DB (Camisassa et al. 2017)
    if normal_mass_model == 'Camisassa2017' and spec_type == 'DB':
        for mass in ['051','054','058','066','074','087','100']:
            if int(mass)/100 < mass_separation_2:
                Cool = Table.read('models/Camisassa_2017_DB_CO/Z002/' + mass + 'DB.trk',
                                  format='ascii') 
                Cool = Cool[(Cool['LOG(TEFF)'] > tmin) * (Cool['LOG(TEFF)'] < tmax)][::1]
                #Cool.sort('Log(edad/Myr)')
                mass_array  = np.concatenate(( mass_array, np.ones(len(Cool)) * int(mass) / 100 ))
                logg        = np.concatenate(( logg, smooth(np.array(Cool['Log(grav)'])) ))
                age         = np.concatenate(( age, 
                                              (10**smooth(np.array(Cool['Log(edad/Myr)'])) - \
                                               10**Cool['Log(edad/Myr)'][0]) * 1e6 + \
                                               (IFMR(int(mass)/100))**(t_index) * 1e10 ))
                age_cool    = np.concatenate(( age_cool, 
                                               (10**smooth(np.array(Cool['Log(edad/Myr)'])) - \
                                                10**Cool['Log(edad/Myr)'][0]) * 1e6 ))
                logteff     = np.concatenate(( logteff, smooth(np.array(Cool['LOG(TEFF)'])) ))
                Mbol        = np.concatenate(( Mbol, 
                                               4.75 - 2.5 * smooth(np.array(Cool['LOG(L)'])) ))
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
    
    # BaSTI model
    for mass in ['054','055','061','068','077','087','100','110','120']:
        if ( normal_mass_model == 'BaSTI' and int(mass)/100 < mass_separation_2 ) or \
           ( high_mass_model == 'BaSTI' and int(mass)/100 > mass_separation_2 ):
            Cool = Table.read('models/BaSTI/'+'COOL'+mass+'BaSTIfinale'+spec_suffix2+'sep.sdss',
                              format='ascii') 
            Cool = Cool[(Cool['log(Teff)'] > tmin) * (Cool['log(Teff)'] < tmax)][::1]#len(Cool)//100
            #Cool.sort('Log(edad/Myr)')
            mass_array  = np.concatenate(( mass_array, np.ones(len(Cool)) * int(mass) / 100 ))
            Cool['Log(grav)'] \
                        = logg_func( np.array(Cool['log(Teff)']), np.ones(len(Cool))*int(mass)/100 )
            logg        = np.concatenate(( logg, smooth(np.array(Cool['Log(grav)'])) ))
            age         = np.concatenate(( age, 10**smooth(np.array(Cool['log(t)'])) - \
                                                10**Cool['log(t)'][0] + \
                                                (IFMR(int(mass)/100))**(t_index) * 1e10 ))
            age_cool    = np.concatenate(( age_cool, 10**smooth(np.array(Cool['log(t)'])) - \
                                                     10**Cool['log(t)'][0]  ))
            logteff     = np.concatenate(( logteff, smooth(np.array(Cool['log(Teff)'])) ))
            Mbol        = np.concatenate(( Mbol, 4.75 - 2.5 * smooth(np.array(Cool['log(L/Lo)'])) ))
    
    # BaSTI high-alpha model
    if 'BaSTI_' in normal_mass_model:
        for time in ['200','300','400','500','600','700','800','900','1000',
                    '1250','1500','1750','2000','2250','2500','2750','3000',
                    '3250','3500','3750','4000','4250','4500','4750','5000',
                    '5250','5500','5750','6000','6250','6500','6750','7000',
                    '7250','7500','7750','8000','8250','8500','8750','9000',
                    '9500','10000','10500','11000','11500','12000','12500','13000','13500','14000']:
            if '4' in normal_mass_model:
                Cool = Table.read('models/BaSTI_z42aeo/WDz402y303aenot' + time + '.' + \
                                  spec_suffix2 + 'sep.sdss', format='ascii') 
            if '2' in normal_mass_model:
                Cool = Table.read('models/BaSTI_z22aeo/WDzsunysunaenot' + time + '.' + \
                                  spec_suffix2 + 'sep.sdss', format='ascii') 
            Cool = Cool[(Cool['log(Teff)'] > tmin) * (Cool['log(Teff)'] < tmax)][::1]#len(Cool)//100
            #Cool.sort('Log(edad/Myr)')
            mass_array  = np.concatenate(( mass_array, np.array(Cool['Mwd']) ))
            Cool['Log(grav)'] = logg_func( np.array(Cool['log(Teff)']), np.array(Cool['Mwd']) )
            logg        = np.concatenate(( logg, smooth(np.array(Cool['Log(grav)'])) ))
            age         = np.concatenate(( age, (np.ones(len(Cool)) * int(time) * 1e6) + \
                                                 IFMR(np.array(Cool['Mwd']))**(t_index) * 1e10 ))
            age_cool    = np.concatenate(( age_cool, (np.ones(len(Cool)) * int(time) * 1e6)   ))
            logteff     = np.concatenate(( logteff, smooth(np.array(Cool['log(Teff)'])) ))
            Mbol        = np.concatenate(( Mbol, 4.75 - 2.5 * smooth(np.array(Cool['log(L/Lo)'])) ))
    
    # PG cooling tracks (Althaus et al. 2009)
    if normal_mass_model == 'PG' and spec_type == 'DB':
        for mass in ['0514','0530','0542','0565','0584','0609','0664','0741','0869']:
            Cool = Table.read('models/tracksPG-DB/db-pg-'+mass+'.trk.t0.11', format='ascii',
                              comment='#')
            Cool = Cool[(Cool['Log(Teff)']>tmin)*(Cool['Log(Teff)']<tmax)*(Cool['age[Myr]']>0)]
            mass_array  = np.concatenate(( mass_array, np.ones(len(Cool))*int(mass)/1000 ))
            logg        = np.concatenate(( logg, smooth(np.array(Cool['Log(grav)'])) ))
            age         = np.concatenate(( age, np.array(Cool['age[Myr]'])*10**6 + \
                                                (IFMR(int(mass)/1000))**(t_index) * 1e10 ))
            age_cool    = np.concatenate(( age_cool, np.array(Cool['age[Myr]']) * 1e6 ))
            logteff     = np.concatenate(( logteff, smooth(np.array(Cool['Log(Teff)'])) ))
            Mbol        = np.concatenate(( Mbol, 4.75 - 2.5 * smooth(np.array(Cool['Log(L)'])) ))
    
    # Ultra-massive ONe model (Camisassa et al. 2019)
    if high_mass_model == 'ONe':
        for mass in ['110','116','122','129']:
            Cool = Table.read('models/ONeWDs/' + mass + '_' + spec_suffix2 + '.trk',format='ascii') 
            Cool = Cool[(Cool['LOG(TEFF)'] > tmin) * (Cool['LOG(TEFF)'] < tmax)][::1]
            #Cool.sort('Log(edad/Myr)')
            mass_array  = np.concatenate(( mass_array, np.ones(len(Cool)) * int(mass) / 100 ))
            logg        = np.concatenate(( logg, smooth(np.array(Cool['Log(grav)'])) ))
            age         = np.concatenate(( age, (10**smooth(np.array(Cool['Log(edad/Myr)'])) - \
                                                 10**Cool['Log(edad/Myr)'][0]) * 1e6 + \
                                                (IFMR(int(mass)/100))**(t_index) * 1e10 ))
            age_cool    = np.concatenate(( age_cool, (10**smooth(np.array(Cool['Log(edad/Myr)'])) - \
                                                      10**Cool['Log(edad/Myr)'][0]) * 1e6 ))
            logteff     = np.concatenate(( logteff, smooth(np.array(Cool['LOG(TEFF)'])) ))
            Mbol        = np.concatenate(( Mbol, 4.75 - 2.5 * smooth(np.array(Cool['LOG(L)'])) ))
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
    
    # massive MESA model (Lauffer et al. 2019)
    if high_mass_model == 'MESA': 
        if spec_suffix3 == 'He':
            mesa_masslist = ['1.0124','1.019','1.0241','1.0358','1.0645','1.0877',
                             '1.1102','1.1254','1.1313','1.1322','1.1466','1.151',
                             '1.2163','1.22','1.2671','1.3075']
        else:
            mesa_masslist = ['1.0124','1.019','1.0241','1.0358','1.0645','1.0877',
                             '1.1102','1.125','1.1309','1.1322','1.1466','1.151',
                             '1.2163','1.22','1.2671','1.3075']
        for mass in mesa_masslist:
            Cool = Table.read('models/MESA_model/' + spec_suffix3 + '_atm-M' + mass + '.dat',
                              format='csv', header_start=1, data_start=2) 
            Cool = Cool[(Cool['# log Teff [K]'] > tmin) * (Cool['# log Teff [K]'] < tmax)][::1]
            mass_array  = np.concatenate(( mass_array, np.ones(len(Cool)) * float(mass) ))
            logg        = np.concatenate(( logg, np.array(Cool['log g [cm/s^2]']) ))
            age         = np.concatenate(( age, Cool['total age [Gyr]'] * 1e9 ))
            age_cool    = np.concatenate(( age_cool, Cool['cooling age [Gyr]'] * 1e9))
            logteff     = np.concatenate(( logteff, np.array(Cool['# log Teff [K]']) ))
            Mbol        = np.concatenate(( Mbol, 4.75 - 2.5 * np.array(Cool['log L/Lsun']) ))
            # additional
            mass_accurate=Cool['mass [Msun]']
            logr        = Cool['log radius [Rsun]']
    
    select = ~np.isnan(mass_array + logg + age + age_cool + logteff + Mbol) * (age_cool > 1e3)
    return mass_array[select], logg[select], age[select], age_cool[select], logteff[select], \
           Mbol[select]


def HR_to_para(bp_rp, G, z, age, HR_grid=(-0.6, 1.5, 0.002, 10, 15, 0.01), interp_type='linear'):
    '''
    Interpolate the mapping of (BR-RP, G) --> z, based on the data points from many 
    cooling tracks read from a model, and get the value of z on the grid of H-R coordinates.
    We set select only G<16 and G>8 to avoid the turning of DA cooling track which
    leads to multi-value mapping.
    
    Arguments:
    bp_rp:    1d-array. The Gaia color BP-RP
    G:        1d-array. The absolute magnitude of Gaia G band
    z:        1d-array. The target parameter for mapping (BP-RP, G) --> z
    HR_grid:  in the form of (xmin, xmax, dx, ymin, ymax, dy), the grid information of 
              the H-R diagram coordinates BP-RP and G
    age:      1d-array. The WD age. Only used for the purpose of selecting non-NaN data points.
    '''
    # define the grid of H-R diagram
    grid_x, grid_y = np.mgrid[HR_grid[0]:HR_grid[1]:HR_grid[2], HR_grid[3]:HR_grid[4]:HR_grid[5]]
    grid_x *= interp_bprp_factor
    
    # select only not-NaN data points
    selected    = ~np.isnan(bp_rp + G + age + z) * (G < 16) * (G > 8)
    
    # get the value of z on a H-R diagram grid and the interpolated mapping
    grid_z      = griddata(np.array((bp_rp[selected]*interp_bprp_factor, G[selected])).T, 
                           z[selected], (grid_x, grid_y), method=interp_type)
    grid_z_func = interpolate_2d(bp_rp[selected], G[selected], z[selected], interp_type)
    
    # return both the grid data and interpolated mapping
    return grid_z, grid_z_func


def interp_xy_z(x, y, z, xy_grid, xfactor=1, interp_type='linear'):
    '''
    Interpolate the mapping (x,y) --> z, based on a series of x, y, and z values, and
    get the value of z on the grid of (x,y) coordinates.
    This function is a generalized version of HR_to_para, allowing any x and y values.
    
    Arguments:
    x:        1d-array. The x in the mapping (x,y) --> z
    y:        1d-array. The y in the mapping (x,y) --> z
    z:        1d-array. The target parameter for mapping (x,y) --> z
    xy_grid:  in the form of (xmin, xmax, dx, ymin, ymax, dy), the grid information of 
              x and y
    xfactor:  Number. For balancing the interval of interpolation between x and y.
    '''
    # define the grid of (x,y)
    grid_x, grid_y = np.mgrid[xy_grid[0]:xy_grid[1]:xy_grid[2], xy_grid[3]:xy_grid[4]:xy_grid[5]]
    grid_x *= xfactor
    
    # select only not-NaN data points
    selected = ~np.isnan(x + y + z)
    
    # get the value of z on a (x,y) grid and the interpolated mapping
    grid_z      = griddata(np.array((x[selected]*xfactor, y[selected])).T, z[selected], 
                           (grid_x, grid_y), method=interp_type)
    grid_z_func = interpolate_2d(x[selected], y[selected], z[selected], interp_type)
    
    # return both the grid data and interpolated mapping
    return grid_z, grid_z_func
  

def interp_xy_z_func(x, y, z, interp_type='linear'):
    '''
    Interpolate the mapping (x,y) --> z, based on a series of x, y, and z values. 
    This function is a generalized version of HR_to_para, allowing any x and y values,
    but does not calculate the grid values as HR_to_para and interp_xy_z do.
    
    Arguments:
    x:      1d-array. The Gaia color BP-RP
    y:      1d-array. The absolute magnitude of Gaia G band
    z:      1d-array. The target parameter for mapping (x,y) --> z
    '''
    # select only not-NaN data points
    selected    = ~np.isnan(x + y + z)
    
    # get the interpolated mapping
    grid_z_func = interpolate_2d(x[selected], y[selected], z[selected], interp_type)
    
    # return the interpolated mapping
    return grid_z_func


#----------------------------------------------------------------------------------------------------   
#
#   Define the main function that reads a set of cooling tracks and generate useful mappings
#
#----------------------------------------------------------------------------------------------------   


def read_and_get_mappings(normal_mass_model, high_mass_model, spec_type):
    # make atmosphere grid and mapping: logteff, logg --> bp-rp,  G-Mbol
    grid_G_Mbol, grid_G_Mbol_func = interp_atm(spec_type, 'G_Mbol', 
                                               T_logg_grid=(tmin,tmax,dt,loggmin,loggmax,dlogg))
    grid_bp_rp, grid_bp_rp_func   = interp_atm(spec_type, 'bp_rp', 
                                               T_logg_grid=(tmin,tmax,dt,loggmin,loggmax,dlogg))
    
    
    # get for logg_func BaSTI models
    if 'BaSTI' in normal_mass_model or 'BaSTI' in high_mass_model:
        mass_array_Fontaine2001, logg_Fontaine, _, _ , logteff_Fontaine2001, _ \
                    = open_evolution_tracks('Fontaine2001', 'Fontaine2001', spec_type)
        logg_func   = interp_xy_z_func(x=logteff_Fontaine2001, y=mass_array_Fontaine2001,
                                       z=logg_Fontaine2001)
    else: 
        logg_func   = None
        
    
    # Open Evolution Tracks
    mass_array, logg, age, age_cool, logteff, Mbol = open_evolution_tracks(normal_mass_model, 
                                                                           high_mass_model,
                                                                           spec_type, logg_func)
    

    # Get Colour/Magnitude for Evolution Tracks
    G       = grid_G_Mbol_func(logteff, logg) + Mbol
    bp_rp   = grid_bp_rp_func(logteff, logg)
    
    
    # Calculate Cooling Rate (per BP-RP)
    k1          = (age_cool[1:-1] - age_cool[:-2]) / (bp_rp[1:-1] - bp_rp[:-2])
    k2          = (age_cool[2:] - age_cool[1:-1]) / (bp_rp[2:] - bp_rp[1:-1])
    k           = k1 + (bp_rp[1:-1] - bp_rp[:-2])*(k1-k2) / (bp_rp[:-2]-bp_rp[2:])
    cool_rate   = np.concatenate(( np.array([1]), k , np.array([1]) ))
    
    
    # Get Parameters on HR Diagram
    grid_mass, grid_mass_func             = HR_to_para( bp_rp, G, mass_array, age, HR_grid )
    grid_logg, grid_logg_func             = HR_to_para( bp_rp, G, logg, age, HR_grid  )
    grid_age, grid_age_func               = HR_to_para( bp_rp, G, age, age, HR_grid )
    grid_age_cool, grid_age_cool_func     = HR_to_para( bp_rp, G, age_cool, age, HR_grid )
    grid_logteff, grid_logteff_func       = HR_to_para( bp_rp, G, logteff, age, HR_grid )
    grid_Mbol, grid_Mbol_func             = HR_to_para( bp_rp, G, Mbol, age, HR_grid )
#     row,col = grid_mass.shape
#     grid_mass_density                     = np.concatenate((np.zeros((row,1)),
#                                                             grid_mass[:,2:] - grid_mass[:,:-2],
#                                                             np.zeros((row,1)) ), axis=1)
    grid_cool_rate, grid_cool_rate_func   = HR_to_para( bp_rp, G, cool_rate, age, HR_grid )
    # (mass, t_cool) --> bp-rp, G
    grid_m_agecool_bprp_func              = interp_xy_z_func( mass_array, age_cool, bp_rp )
    grid_m_agecool_G_func                 = interp_xy_z_func( mass_array, age_cool, G )
    
    
    # Return a dictionary containing all the cooling track data points, interpolation functions
    # and interpolation grids 
    return {'grid_G_Mbol':grid_G_Mbol, 'grid_G_Mbol_func':grid_G_Mbol_func,
            'grid_bp_rp':grid_bp_rp, 'grid_bp_rp_func':grid_bp_rp_func,
            'mass_array':mass_array, 'logg':logg, 'age':age, 'age_cool':age_cool,
            'logteff':logteff, 'Mbol':Mbol, 'G':G, 'bp_rp':bp_rp, 'cool_rate':cool_rate,
            'grid_mass':grid_mass, 'grid_mass_func':grid_mass_func,
            'grid_logg':grid_logg, 'grid_logg_func':grid_logg_func,
            'grid_age':grid_age, 'grid_age_func':grid_age_func,
            'grid_age_cool':grid_age_cool, 'grid_age_cool_func':grid_age_cool_func,
            'grid_logteff':grid_logteff, 'grid_logteff_func':grid_logteff_func,
            'grid_Mbol':grid_Mbol, 'grid_Mbol_func':grid_Mbol_func,
            'grid_cool_rate':grid_cool_rate, 'grid_cool_rate_func':grid_cool_rate_func,
            'grid_m_agecool_bprp_func':grid_m_agecool_bprp_func, 
            'grid_m_agecool_G_func':grid_m_agecool_G_func}


#----------------------------------------------------------------------------------------------------   
#
#   Defind
#
#----------------------------------------------------------------------------------------------------  


tmin = 3.5; tmax = 5.1; dt = 0.01
loggmin = 6.5; loggmax = 9.6; dlogg = 0.01
HR_grid             = (-0.6, 1.5, 0.002, 10, 15, 0.01) # bp_rp, G
interp_type         = 'linear'
interp_type_atm     = 'linear'
interp_bprp_factor  = 5

# Fontaine et al. 2001 (CO), Camisassa et al. 2019 (ONe), PG, and Lauffer et al. 2019 (MESA) models
DA_thick_CO     = main('Fontaine2001', 'Fontaine2001', 'DA_thick')
DB_CO           = main('Fontaine2001', 'Fontaine2001', 'DB')
DA_thick_ONe    = main('Fontaine2001', 'ONe', 'DA_thick')
DB_ONe          = main('Fontaine2001', 'ONe', 'DB')
DA_thick_LPONe  = main('Althaus2010', 'ONe', 'DA_thick')
DB_LPONe        = main('Camisassa2017', 'ONe', 'DB')
DB_PGONe        = main('PG', 'ONe', 'DB')
DA_thick_MESA   = main('Fontaine2001', 'MESA', 'DA_thick')
DB_MESA         = main('Fontaine2001', 'MESA', 'DB')

# Salaris et al. 2010 (Phase_Sep) BaSTI models. 4 and 2 are alpha-enhanced models.
DA_thick_Phase_Sep  = main('BasTI', 'ONe', 'DA_thick')
DB_Phase_Sep        = main('BasTI', 'ONe', 'DB')

DA_thick_Phase_Sep_4= main('BasTI_4', 'ONe', 'DA_thick')
DB_Phase_Sep_4      = main('BasTI_4', 'ONe', 'DB')

DA_thick_Phase_Sep_2= main('BasTI_2', 'ONe', 'DA_thick')
DB_Phase_Sep_2      = main('BasTI_2', 'ONe', 'DB')
