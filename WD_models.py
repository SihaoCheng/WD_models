'''
This package will read different cooling models and interpolate the conversion functions
between HR diagram, Teff, Mbol, etc. The functions are stored in dictionaries for each model.
See the main function and the lines after its definition.
This package also contains the functions to read a single cooling track.
'''


import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, vstack, hstack
from scipy.interpolate import interp1d, interp2d, CloughTocher2DInterpolator, griddata, LinearNDInterpolator


#----------------------------------------------------------------------------------------------------   

IFMR = interp1d((0.2,0.3,0.4,0.50, 0.55, 0.65,0.75, 0.85, 1.0, 1.25,1.35),(0.3,0.5,0.7,0.95,1,2,3,3.5,5,8,9),\
                          fill_value = 0, bounds_error=False) # mass_WD, mass_ini
t_index = -3

#----------------------------------------------------------------------------------------------------   

def interpolate_2d(x,y,para,method):
    if method == 'linear':
        interpolator = LinearNDInterpolator
    elif method == 'cubic':
        interpolator = CloughTocher2DInterpolator
    return interpolator((x,y), para, rescale=True)
  

def interp_atm(spec_type, color, xy):
    '''
    This function generates from the atmosphere model the function (logTeff, logg) --> G, BP-RP, or G-Mbol 
    
    Arguments:
    spec_type: 'DA_thick' or 'DA_thin' or 'DB'. See http://www.astro.umontreal.ca/~bergeron/CoolingModels/
    color:     the target variable of the interpolation function. 'G', 'BP-RP', or 'G-Mbol'.
    xy:        in the form (xmin, xmax, dx, ymin, ymax, dy), corresponding to the grid of logTeff and logg.
    '''
    xy=(tmin,tmax,dt,loggmin,loggmax,dlogg)
    logTeff     = np.zeros(0)
    logg        = np.zeros(0)
    age         = np.zeros(0)
    mass_array  = np.zeros(0)
    G           = np.zeros(0)
    bp_rp       = np.zeros(0)
    Mbol        = np.zeros(0)
    
    # read the table for each mass
    for mass in ['0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0','1.2']:
        Cool    = Table.read('models/Fontaine_Gaia_atm_grid/Table_Mass_'+mass+'_'+spec_type,format='ascii')
        selected= Cool['Teff']>3500
        Cool    = Cool[selected]
        
        # read columns
        bp_rp   = np.concatenate((bp_rp,Cool['G_BP/R']-Cool['G_RP/R']))
        G       = np.concatenate((G,Cool['G/R']))
        Mbol    = np.concatenate((Mbol,Cool['Mbol']))
        logTeff = np.concatenate((logTeff,np.log10(Cool['Teff'])))
        logg    = np.concatenate((logg,Cool['logg']))
        age     = np.concatenate((age,Cool['Age']))
        
    # read the table for all logg
    Cool    = Table.read('models/Fontaine_Gaia_atm_grid/Table_'+spec_type,format='ascii') # the whole grid
    selected= Cool['Teff']>3500
    Cool    = Cool[selected]
    
    # read columns
    bp_rp   = np.concatenate((bp_rp,Cool['G_BP/R']-Cool['G_RP/R']))
    G       = np.concatenate((G,Cool['G/R']))
    Mbol    = np.concatenate((Mbol,Cool['Mbol']))
    logTeff = np.concatenate((logTeff,np.log10(Cool['Teff'])))
    logg    = np.concatenate((logg,Cool['logg']))
    age     = np.concatenate((age,Cool['Age']))        
    
    grid_x, grid_y = np.mgrid[xy[0]:xy[1]:xy[2], xy[3]:xy[4]:xy[5]]
    
    # define the interpolation function
    def interp(x,y,z):
        grid_z      = griddata(np.array((x,y)).T, z, (grid_x, grid_y), method=interp_type_atm)
        grid_z_func = interpolate_2d(x, y, z, interp_type_atm)
        return grid_z, grid_z_func
    
    if color == 'G':
        return interp(logTeff,logg,G)
    if color == 'bp_rp':
        return interp(logTeff,logg,bp_rp)
    if color == 'G_Mbol':
        return interp(logTeff,logg,G-Mbol)


def HR_to_para(z, xy, bp_rp, G, age):
    '''
    Interpolate the function of (BR-RP, G) --> z, based on the data points from many 
    cooling tracks read from a model, and get the value of z on the grid of H-R coordinates.
    We set select only G<16 and G>8 to avoid the turning of DA cooling track which
    leads to multi-value mapping.
    
    Arguments:
    z:      1d-array. The target parameter for mapping (BP-RP, G) --> z
    xy:     in the form of (xmin, xmax, dx, ymin, ymax, dy), the grid information of 
            the H-R diagram coordinates BP-RP and G
    bp_rp:  1d-array. The Gaia color BP-RP
    G:      1d-array. The absolute magnitude of Gaia G band
    age:    1d-array. The WD age. Only used for the purpose of selecting non-NaN data points.
    '''
    # define the grid of H-R diagram
    grid_x, grid_y = np.mgrid[xy[0]:xy[1]:xy[2], xy[3]:xy[4]:xy[5]]
    grid_x *= interp_bprp_factor
    
    # select only not-NaN data points
    selected    = ~np.isnan(bp_rp+G+age+z)*(G<16)*(G>8)
    
    # get the value of z on a H-R diagram grid and the interpolated function
    grid_z      = griddata(np.array((bp_rp[selected]*interp_bprp_factor, G[selected])).T, z[selected],
                           (grid_x, grid_y), method=interp_type)
    grid_z_func = interpolate_2d(bp_rp[selected], G[selected], z[selected], interp_type)
    
    # return both the grid data and interpolated function
    return grid_z, grid_z_func


def interp_xy_z(z, xy, x, y, xfactor=1):
    '''
    Interpolate the function (x,y) --> z, based on a series of x, y, and z values, and
    get the value of z on the grid of (x,y) coordinates.
    This function is a generalized version of HR_to_para, allowing any x and y values.
    
    Arguments:
    z:      1d-array. The target parameter for mapping (x,y) --> z
    xy:     in the form of (xmin, xmax, dx, ymin, ymax, dy), the grid information of 
            x and y
    x:      1d-array. The Gaia color BP-RP
    y:      1d-array. The absolute magnitude of Gaia G band
    xfactor:Number. For balancing the interval of interpolation between x and y.
    '''
    # define the grid of (x,y)
    grid_x, grid_y = np.mgrid[xy[0]:xy[1]:xy[2], xy[3]:xy[4]:xy[5]]
    grid_x *= xfactor
    
    # select only not-NaN data points
    selected = ~np.isnan(x+y+z)
    
    # get the value of z on a (x,y) grid and the interpolated function
    grid_z      = griddata(np.array((x[selected]*xfactor, y[selected])).T, z[selected], 
                           (grid_x, grid_y), method=interp_type)
    grid_z_func = interpolate_2d(x[selected], y[selected], z[selected], interp_type)
    
    # return both the grid data and interpolated function
    return grid_z, grid_z_func
  

def interp_xy_z_func(z,x,y):
    '''
    Interpolate the function (x,y) --> z, based on a series of x, y, and z values. 
    This function is a generalized version of HR_to_para, allowing any x and y values,
    but does not calculate the grid values as HR_to_para and interp_xy_z do.
    
    Arguments:
    z:      1d-array. The target parameter for mapping (x,y) --> z
    x:      1d-array. The Gaia color BP-RP
    y:      1d-array. The absolute magnitude of Gaia G band
    '''
    # select only not-NaN data points
    selected    = ~np.isnan(x+y+z)
    
    # get the interpolated function
    grid_z_func = interpolate_2d(x[selected], y[selected], z[selected], interp_type)
    
    # return the interpolated function
    return grid_z_func


def open_evolution_tracks(model, spec_type, logg_func=None):
    '''
    Read the cooling models and store the following information of different cooling
    tracks together in one numpy array: mass, logg, age, age_cool, logteff, Mbol
    '''
    logg        = np.zeros(0)
    age         = np.zeros(0) 
    age_cool    = np.zeros(0)
    logteff     = np.zeros(0)
    mass_array  = np.zeros(0)
    Mbol        = np.zeros(0)
    
    # the list of cooling tracks to read
    CO          = ['020','030','040','050','060','070','080','090','095','100','105',
                   '110','115','120','125','130']
    ONe         = ['020','030','040','050','060','070','080','090','095','100','105']
    MESA        = ['020','030','040','050','060','070','080','090','095']
    Phase_Sep   = ['054','055','061','068','077','087','100','110','120']
    if model == 'CO':
        mass_list = CO
    if model == 'CO+ONe':
        mass_list = ONe
    if model == 'PG+ONe' or model == 'PG+MESA' or ('Phase_Sep' in model):
        mass_list = []
    if model == 'CO+MESA':
        mass_list = MESA
    if model == 'Phase_Sep+ONe':
        Phase_Sep = ['054','055','061','068','077','087','100']
    
    # define some 
    if spec_type == 'DB':
        spec_suffix = '0210'; spec_suffix2 = 'DB'; spec_suffix3 = 'He'
    else:
        spec_suffix = '0204'; spec_suffix2 = 'DA'; spec_suffix3 = 'H'
    
    # read cooling tracks
    # Fontaine et al. 2001
    for mass in mass_list:
        f       = open('models/Fontaine_AllSequences/CO_'+mass+spec_suffix)
        text    = f.read()
        example = "      1    57674.0025    8.36722799  7.160654E+08  4.000000E+05  4.042436E+33\n" + \
                  "        7.959696E+00  2.425570E+01  7.231926E+00  0.0000000000  0.000000E+00\n" + \
                  "        6.019629E+34 -4.010597E+00 -1.991404E+00 -3.055254E-01 -3.055254E-01"
        logg_temp       = []
        age_temp        = []
        age_cool_temp   = []
        logteff_temp    = []
        Mbol_temp       = []
        for line in range(len(text)//len(example)):
            logteff_temp.append( np.log10(float(text[line*len(example)+9:line*len(example)+21])))
            logg_temp.append( float(text[line*len(example)+22:line*len(example)+35]))
            age_temp.append( float(text[line*len(example)+48:line*len(example)+63]) +\
                            (IFMR(int(mass)/100))**(t_index) * 1e10)
            age_cool_temp.append( float(text[line*len(example)+48:line*len(example)+63]) )
            Mbol_temp.append( 4.75 - 2.5 * np.log10(float(text[line*len(example)+64:line*len(example)+76]) / 3.828e33) )
        mass_array  = np.concatenate((mass_array, np.ones(len(logg_temp))*int(mass)/100))
        logg        = np.concatenate((logg, logg_temp))
        age         = np.concatenate((age, age_temp))
        age_cool    = np.concatenate((age_cool, age_cool_temp))
        logteff     = np.concatenate((logteff, logteff_temp))
        Mbol        = np.concatenate((Mbol, Mbol_temp))
    
    def smooth(x,window_len=5,window='hanning'):
        w=eval('np.'+window+'(window_len)')
        y=np.convolve(w/w.sum(),x,mode='same')
        return x
    
    # BaSTI model
    if model == 'Phase_Sep' or model == 'Phase_Sep+ONe':
        for mass in Phase_Sep:
            Cool = Table.read('models/BaSTI/'+'COOL'+mass+'BaSTIfinale'+spec_suffix2+'sep.sdss', format='ascii') 
            Cool = Cool[(Cool['log(Teff)']>tmin)*(Cool['log(Teff)']<tmax)][::len(Cool)//100]
            #Cool.sort('Log(edad/Myr)')
            mass_array  = np.concatenate(( mass_array, np.ones(len(Cool))*int(mass)/100 ))
            Cool['Log(grav)'] = logg_func( np.array(Cool['log(Teff)']), np.ones(len(Cool))*int(mass)/100 )
            logg        = np.concatenate(( logg, smooth(np.array(Cool['Log(grav)'])) ))
            age         = np.concatenate(( age, 10**smooth(np.array(Cool['log(t)'])) - 10**Cool['log(t)'][0] + \
                                                (IFMR(int(mass)/100))**(t_index) * 1e10 ))
            age_cool    = np.concatenate(( age_cool, 10**smooth(np.array(Cool['log(t)'])) - 10**Cool['log(t)'][0]  ))
            logteff     = np.concatenate(( logteff, smooth(np.array(Cool['log(Teff)'])) ))
            Mbol        = np.concatenate(( Mbol, 4.75 - 2.5 * smooth(np.array(Cool['log(L/Lo)'])) ))
    
    # BaSTI high-alpha model
    if 'Phase_Sep_' in model:
        for time in ['200','300','400','500','600','700','800','900','1000',
                    '1250','1500','1750','2000','2250','2500','2750','3000',
                    '3250','3500','3750','4000','4250','4500','4750','5000',
                    '5250','5500','5750','6000','6250','6500','6750','7000',
                    '7250','7500','7750','8000','8250','8500','8750','9000',
                    '9500','10000','10500','11000','11500','12000','12500','13000','13500','14000']:
            if '4' in model:
                Cool = Table.read('models/BaSTI_z42aeo/WDz402y303aenot'+time+'.'+spec_suffix2+'sep.sdss', format='ascii') 
            if '2' in model:
                Cool = Table.read('models/BaSTI_z22aeo/WDzsunysunaenot'+time+'.'+spec_suffix2+'sep.sdss', format='ascii') 
            Cool = Cool[(Cool['log(Teff)']>tmin)*(Cool['log(Teff)']<tmax)][::len(Cool)//100]
            #Cool.sort('Log(edad/Myr)')
            mass_array  = np.concatenate(( mass_array, np.array(Cool['Mwd']) ))
            Cool['Log(grav)'] = logg_func( np.array(Cool['log(Teff)']), np.array(Cool['Mwd']) )
            logg        = np.concatenate(( logg, smooth(np.array(Cool['Log(grav)'])) ))
            age         = np.concatenate(( age, (np.ones(len(Cool)) * int(time) * 1e6) + \
                                                IFMR(np.array(Cool['Mwd']))**(t_index) * 1e10 ))
            age_cool    = np.concatenate(( age_cool, (np.ones(len(Cool)) * int(time) * 1e6)   ))
            logteff     = np.concatenate(( logteff, smooth(np.array(Cool['log(Teff)'])) ))
            Mbol        = np.concatenate(( Mbol, 4.75 - 2.5 * smooth(np.array(Cool['log(L/Lo)'])) ))
    
    # read PG cooling tracks
    if model == 'PG+ONe':
        for mass in ['0514','0530','0542','0565','0584','0609','0664','0741','0869']:
            Cool = Table.read('models/tracksPG-DB/db-pg-'+mass+'.trk.t0.11', format='ascii', comment='#')
            Cool = Cool[(Cool['Log(Teff)']>tmin)*(Cool['Log(Teff)']<tmax)*(Cool['age[Myr]']>0)]
            mass_array  = np.concatenate(( mass_array, np.ones(len(Cool))*int(mass)/1000 ))
            logg        = np.concatenate(( logg, smooth(np.array(Cool['Log(grav)'])) ))
            age         = np.concatenate(( age, np.array(Cool['age[Myr]'])*10**6 + \
                                                (IFMR(int(mass)/1000))**(t_index) * 1e10 ))
            age_cool    = np.concatenate(( age_cool, np.array(Cool['age[Myr]']) * 1e6 ))
            logteff     = np.concatenate(( logteff, smooth(np.array(Cool['Log(Teff)'])) ))
            Mbol        = np.concatenate(( Mbol, 4.75 - 2.5 * smooth(np.array(Cool['Log(L)'])) ))
            
    # read ultra-massive ONe model (Camisassa et al. 2019)
    if model == 'CO+ONe' or model == 'PG+ONe' or model == 'Phase_Sep+ONe' or ('Phase_Sep_' in model): 
        for mass in ['110','116','122','129']:
            Cool = Table.read('models/ONeWDs/'+mass+'_'+spec_suffix2+'.trk',format='ascii') 
            Cool = Cool[(Cool['LOG(TEFF)']>tmin)*(Cool['LOG(TEFF)']<tmax)][::10]
            Cool.sort('Log(edad/Myr)')
            mass_array  = np.concatenate(( mass_array, np.ones(len(Cool))*int(mass)/100 ))
            logg        = np.concatenate(( logg, smooth(np.array(Cool['Log(grav)'])) ))
            age         = np.concatenate(( age, (10**smooth(np.array(Cool['Log(edad/Myr)'])) - \
                                                 10**Cool['Log(edad/Myr)'][0]) * 1e6 + \
                                                (IFMR(int(mass)/100))**(t_index) * 1e10 ))
            age_cool    = np.concatenate(( age_cool, (10**smooth(np.array(Cool['Log(edad/Myr)'])) - \
                                                      10**Cool['Log(edad/Myr)'][0]) * 1e6 ))
            logteff     = np.concatenate(( logteff, smooth(np.array(Cool['LOG(TEFF)'])) ))
            Mbol        = np.concatenate(( Mbol, 4.75 - 2.5 * smooth(np.array(Cool['LOG(L)'])) ))
            
    # read massive MESA model
    if model=='CO+MESA' or model=='PG+MESA': 
        if spec_suffix3 == 'He':
            mesa_masslist = ['1.0124','1.019','1.0241','1.0358','1.0645','1.0877','1.1102','1.1254','1.1313','1.1322',\
                     '1.1466','1.151','1.2163','1.22','1.2671','1.3075']
        else:
            mesa_masslist = ['1.0124','1.019','1.0241','1.0358','1.0645','1.0877','1.1102','1.125','1.1309','1.1322',\
                     '1.1466','1.151','1.2163','1.22','1.2671','1.3075']
        for mass in mesa_masslist:
            Cool = Table.read('models/MESA_model/'+spec_suffix3+'_atm-M'+mass+'.dat', format='csv',
                              header_start=1, data_start=2) 
            Cool = Cool[(Cool['# log Teff [K]']>tmin)*(Cool['# log Teff [K]']<tmax)][::1]
            #Cool.sort('Log(edad/Myr)')
            mass_array  = np.concatenate(( mass_array, Cool['mass [Msun]'] ))
            logg        = np.concatenate(( logg, np.array(Cool['log g [cm/s^2]']) ))
            age         = np.concatenate(( age, Cool['total age [Gyr]'] * 1e9 ))
            age_cool    = np.concatenate(( age_cool, Cool['cooling age [Gyr]'] * 1e9))
            logteff     = np.concatenate(( logteff, np.array(Cool['# log Teff [K]']) ))
            Mbol        = np.concatenate(( Mbol, 4.75 - 2.5 * np.array(Cool['log L/Lsun']) ))
    
    select = ~np.isnan(mass_array + logg + age + age_cool + logteff + Mbol) * (np.log10(age_cool) > -10)
    return mass_array[select], logg[select], age[select], age_cool[select], logteff[select], Mbol[select]


#----------------------------------------------------------------------------------------------------   


#----------------------------------------------------------------------------------------------------   

def main(spec_type, model, logg_func=None):
    # Make Atmosphere Grid/Function: logTeff, logg --> bp-rp,  G-Mbol
    grid_G_Mbol, grid_G_Mbol_func   = interp_atm(spec_type, 'G_Mbol', xy=(tmin,tmax,dt,loggmin,loggmax,dlogg))
    grid_bp_rp, grid_bp_rp_func     = interp_atm(spec_type, 'bp_rp', xy=(tmin,tmax,dt,loggmin,loggmax,dlogg))
    
    
    # Open Evolution Tracks
    mass_array, logg, age, age_cool, logteff, Mbol = open_evolution_tracks(model, spec_type, logg_func)
    

    # Get Colour/Magnitude for Evolution Tracks
    G       = grid_G_Mbol_func(logteff, logg) + Mbol
    bp_rp   = grid_bp_rp_func(logteff, logg)
    
    
    # Calculate Cooling Rate (per BP-RP)
    k1          = (age_cool[1:-1] - age_cool[:-2]) / (bp_rp[1:-1] - bp_rp[:-2])
    k2          = (age_cool[2:] - age_cool[1:-1]) / (bp_rp[2:] - bp_rp[1:-1])
    k           = k1 + (bp_rp[1:-1] - bp_rp[:-2])*(k1-k2) / (bp_rp[:-2]-bp_rp[2:])
    cool_rate   = np.concatenate(( np.array([1]), k , np.array([1]) ))
    
    
    # Get Parameters on HR Diagram
    grid_mass, grid_mass_func               = HR_to_para( mass_array, xy, bp_rp, G, age )
    grid_logg, grid_logg_func               = HR_to_para( logg, xy, bp_rp, G,age )
    grid_logage, grid_logage_func           = HR_to_para( np.log10(age), xy, bp_rp, G, age )
    grid_logage_cool, grid_logage_cool_func = HR_to_para( np.log10(age_cool), xy, bp_rp, G, age )
    grid_teff, grid_teff_func               = HR_to_para( 10**logteff, xy, bp_rp, G,age )
    grid_Mbol, grid_Mbol_func               = HR_to_para( Mbol, xy, bp_rp, G, age )
    row,col = grid_mass.shape
    grid_mass_density                       = np.concatenate((np.zeros((row,1)),
                                                              grid_mass[:,2:] - grid_mass[:,:-2],
                                                              np.zeros((row,1)) ), axis=1)
    grid_cool_rate, grid_cool_rate_func     = HR_to_para( cool_rate, xy, bp_rp, G, age )
    # (mass, log(t_cool)) --> bp-rp, G
    grid_bprp_func                          = interp_xy_z_func( bp_rp, mass_array, np.log10(age_cool) )
    grid_G_func                             = interp_xy_z_func( G, mass_array, np.log10(age_cool) )
    
    
    # Return a dictionary containing all the cooling track data points, interpolation functions and interpolation grids 
    return {'grid_G_Mbol':grid_G_Mbol, 'grid_G_Mbol_func':grid_G_Mbol_func,
            'grid_bp_rp':grid_bp_rp, 'grid_bp_rp_func':grid_bp_rp_func,
            'mass_array':mass_array, 'logg':logg, 'age':age, 'age_cool':age_cool,
            'logteff':logteff, 'Mbol':Mbol, 'G':G, 'bp_rp':bp_rp, 'cool_rate':cool_rate,
            'grid_mass':grid_mass, 'grid_mass_func':grid_mass_func,
            'grid_logg':grid_logg, 'grid_logg_func':grid_logg_func,
            'grid_logage':grid_logage, 'grid_logage_func':grid_logage_func,
            'grid_logage_cool':grid_logage_cool, 'grid_logage_cool_func':grid_logage_cool_func,
            'grid_teff':grid_teff, 'grid_teff_func':grid_teff_func,
            'grid_Mbol':grid_Mbol, 'grid_Mbol_func':grid_Mbol_func,
            'grid_cool_rate':grid_cool_rate, 'grid_cool_rate_func':grid_cool_rate_func,
            'grid_mass_density':grid_mass_density,
            'grid_bprp_func':grid_bprp_func, 'grid_G_func':grid_G_func}


#----------------------------------------------------------------------------------------------------   


#----------------------------------------------------------------------------------------------------   


tmin = 3.5; tmax = 5.1; dt = 0.01
loggmin = 6.5; loggmax = 9.6; dlogg = 0.01
xy = (-0.6, 1.5, 0.002, 10, 15, 0.01) # bp_rp, G
interp_type = 'linear'
interp_type_atm = 'linear'
interp_bprp_factor = 5

# Fontaine et al. 2001 (CO), Camisassa et al. 2019 (ONe), PG, and Lauffer et al. 2019 (MESA) models
DA_thick_CO     = main('DA_thick','CO')
DA_thin_CO      = main('DA_thin','CO')
DB_CO           = main('DB','CO')
DA_thick_ONe    = main('DA_thick','CO+ONe')
DA_thin_ONe     = main('DA_thin','CO+ONe')
DB_ONe          = main('DB','CO+ONe')
DB_PGONe        = main('DB','PG+ONe')
DA_thick_MESA   = main('DA_thick','CO+MESA')
DB_MESA         = main('DB','CO+MESA')

# get BaSTI logg_func
logg_func_DA_thick_CO   = interp_xy_z_func(DA_thick_CO['logg'], DA_thick_CO['logteff'], DA_thick_CO['mass_array'])
logg_func_DA_thin_CO    = interp_xy_z_func(DA_thin_CO['logg'], DA_thin_CO['logteff'], DA_thin_CO['mass_array'])
logg_func_DB_CO         = interp_xy_z_func(DB_CO['logg'], DB_CO['logteff'], DB_CO['mass_array'])

# Salaris et al. 2010 (Phase_Sep) BaSTI models. 4 and 2 are alpha-enhanced models.
DA_thick_Phase_Sep  = main('DA_thick','Phase_Sep', logg_func_DA_thick_CO)
DB_Phase_Sep        = main('DB','Phase_Sep', logg_func_DB_CO)

DA_thick_Phase_Sep_4= main('DA_thick','Phase_Sep_4', logg_func_DA_thick_CO)
DB_Phase_Sep_4      = main('DB','Phase_Sep_4', logg_func_DB_CO)

DA_thick_Phase_Sep_2= main('DA_thick','Phase_Sep_2', logg_func_DA_thick_CO)
DB_Phase_Sep_2      = main('DB','Phase_Sep_2', logg_func_DB_CO)


#----------------------------------------------------------------------------------------------------   


#----------------------------------------------------------------------------------------------------   

# Inspect One Evolutionary Track
def open_a_track(spec_type, model, mass, logg_func=None):
    tmin = 2.96; tmax = 5; dt = 0.01
    loggmin = 6.5; loggmax = 9.5; dlogg = 0.01
    
    mass_array  = np.zeros(0)
    logg        = np.zeros(0)
    age         = np.zeros(0)
    age_cool    = np.zeros(0)
    logteff     = np.zeros(0)
    Mbol        = np.zeros(0)

    if spec_type == 'DB':
        spec_suffix = '0210'; spec_suffix2 = 'DB'; spec_suffix3 = 'He'
    else:
        spec_suffix = '0204'; spec_suffix2 = 'DA'; spec_suffix3 = 'H'
        
    if model == 'CO':  
        CO_masslist = ['020','030','040','050','060','070','080','090','095','100','105','110','115','120','125','130']
        mass_diff = 1000; mass_temp=''
        for mass_CO in CO_masslist:
            if abs(float(mass)-float(mass_CO)) < mass_diff:
                mass_diff = abs(float(mass)-float(mass_CO))
                mass_temp = mass_CO
        mass = mass_temp
        
        f       = open('models/Fontaine_AllSequences/CO_'+mass+spec_suffix)
        text    = f.read()
        example = "      1    57674.0025    8.36722799  7.160654E+08  4.000000E+05  4.042436E+33\n" + \
                  "7.959696E+00  2.425570E+01  7.231926E+00  0.0000000000  0.000000E+00\n" + \
                  "6.019629E+34 -4.010597E+00 -1.991404E+00 -3.055254E-01 -3.055254E-01"
        logg_temp       = []
        age_temp        = []
        age_cool_temp   = []
        logteff_temp    = []
        Mbol_temp       = []
        for line in range(len(text)//len(example)):
            logteff_temp.append( np.log10(float(text[line*len(example)+9:line*len(example)+21])))
            logg_temp.append( float(text[line*len(example)+22:line*len(example)+35]))
            age_temp.append( float(text[line*len(example)+48:line*len(example)+63]) +\
                            (IFMR(int(mass)/100))**(t_index) * 1e10)
            age_cool_temp.append( float(text[line*len(example)+48:line*len(example)+63]) )
            Mbol_temp.append( 4.75 - 2.5 * np.log10(float(text[line*len(example)+64:line*len(example)+76])/3.828e33) )
        mass_array  = np.concatenate((mass_array,np.ones(len(logg_temp))*int(mass)/100))
        logg        = np.concatenate((logg,logg_temp))
        age         = np.concatenate((age,age_temp))
        age_cool    = np.concatenate((age_cool,age_cool_temp))
        logteff     = np.concatenate((logteff,logteff_temp))
        Mbol        = np.concatenate((Mbol,Mbol_temp))
        
    if model == 'ONe':
        def smooth(x,window_len=10,window='hanning'):
            s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
            w=eval('np.'+window+'(window_len)')
            y=np.convolve(w/w.sum(),s,mode='same')
            return x
        if spec_type == 'DB':
            spec_suffix2 = 'DB'
        else:
            spec_suffix2 = 'DA'
        Cool = Table.read('models/ONeWDs/'+mass+'_'+spec_suffix2+'.trk',format='ascii') 
        Cool = Cool[(Cool['LOG(TEFF)']>tmin)*(Cool['LOG(TEFF)']<tmax)][::len(Cool)//50]
        Cool.sort('Log(edad/Myr)')
        mass_array  = np.concatenate(( mass_array, np.ones(len(Cool))*int(mass)/100 ))
        logg        = np.concatenate(( logg, smooth(np.array(Cool['Log(grav)'])) ))
        age         = np.concatenate(( age, (10**smooth(np.array(Cool['Log(edad/Myr)'])) - \
                                             10**Cool['Log(edad/Myr)'][0]) * 1e6 ))
        age_cool    = np.concatenate(( age_cool, (10**smooth(np.array(Cool['Log(edad/Myr)'])) - \
                                                  10**Cool['Log(edad/Myr)'][0]) * 1e6 - \
                                                 (IFMR(int(mass)/100))**(t_index) * 1e10 ))
        logteff     = np.concatenate(( logteff, smooth(np.array(Cool['LOG(TEFF)'])) ))
        Mbol        = np.concatenate(( Mbol, 4.75-2.5*smooth(np.array(Cool['LOG(L)'])) ))
        
    if model == 'Phase_Sep':
        def smooth(x,window_len=10,window='hanning'):
            s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
            w=eval('np.'+window+'(window_len)')
            y=np.convolve(w/w.sum(),s,mode='same')
            return x
        if spec_type == 'DB':
            spec_suffix2 = 'DB'
        else:
            spec_suffix2 = 'DA'
        Cool = Table.read('models/BaSTI/'+'COOL'+mass+'BaSTIfinale'+spec_suffix2+'sep.sdss',
                          format='ascii') 
        Cool = Cool[(Cool['log(Teff)']>tmin)*(Cool['log(Teff)']<tmax)][::len(Cool)//100]
        #Cool.sort('Log(edad/Myr)')
        mass_array  = np.concatenate(( mass_array, np.ones(len(Cool))*int(mass)/100 ))
        Cool['Log(grav)'] = logg_func( np.array(Cool['log(Teff)']), np.ones(len(Cool))*int(mass)/100 )
        logg        = np.concatenate(( logg, smooth(np.array(Cool['Log(grav)'])) ))
        age         = np.concatenate(( age, 10**smooth(np.array(Cool['log(t)'])) - 10**Cool['log(t)'][0] + \
                                            (IFMR(int(mass)/100))**(t_index) * 1e10 ))
        age_cool    = np.concatenate(( age_cool, 10**smooth(np.array(Cool['log(t)'])) - 10**Cool['log(t)'][0]  ))
        logteff     = np.concatenate(( logteff, smooth(np.array(Cool['log(Teff)'])) ))
        Mbol        = np.concatenate(( Mbol, 4.75 - 2.5 * smooth(np.array(Cool['log(L/Lo)'])) ))
    
    if model == 'PGONe':
        PG_masslist = ['0514','0530','0542','0565','0584','0609','0664','0741','0869']
        mass_diff = 10000; mass_temp=''
        for mass_PG in PG_masslist:
            if abs(float(mass)-float(mass_PG)) < mass_diff:
                mass_diff = abs(float(mass)-float(mass_PG))
                mass_temp = mass_PG
        mass = mass_temp
        Cool = Table.read('models/tracksPG-DB/db-pg-'+mass+'.trk.t0.11',format='ascii',comment='#')
        Cool = Cool[(Cool['Log(Teff)']>tmin)*(Cool['Log(Teff)']<tmax)*(Cool['age[Myr]']>0)][::len(Cool)//200]
        mass_array  = np.concatenate(( mass_array, np.ones(len(Cool))*int(mass)/1000 ))
        logg        = np.concatenate(( logg, (np.array(Cool['Log(grav)'])) ))
        age         = np.concatenate(( age, np.array(Cool['age[Myr]']) * 1e6 + (IFMR(int(mass)/1000))**(t_index) * 1e10 ))
        age_cool    = np.concatenate(( age_cool, np.array(Cool['age[Myr]']) * 1e6 ))
        logteff     = np.concatenate(( logteff, (np.array(Cool['Log(Teff)'])) ))
        Mbol        = np.concatenate(( Mbol, 4.75 - 2.5 * (np.array(Cool['Log(L)'])) ))
        
    if model == 'MESA':
        if spec_type == 'DB':
            mesa_masslist = ['1.0124','1.019','1.0241','1.0358','1.0645','1.0877','1.1102','1.1254','1.1313','1.1322',\
                     '1.1466','1.151','1.2163','1.22','1.2671','1.3075']
        else:
            mesa_masslist = ['1.0124','1.019','1.0241','1.0358','1.0645','1.0877','1.1102','1.125','1.1309','1.1322',\
                     '1.1466','1.151','1.2163','1.22','1.2671','1.3075']
        mass_diff = 100; mass_temp=''
        for mass_mesa in mesa_masslist:
            if abs(float(mass)/100-float(mass_mesa)) < mass_diff:
                mass_diff = abs(float(mass)/100-float(mass_mesa))
                mass_temp = mass_mesa                
        mass = mass_temp
        Cool = Table.read('models/MESA_model/'+spec_suffix3+'_atm-M'+mass+'.dat',format='csv',header_start=1,data_start=2) 
        Cool = Cool[(Cool['# log Teff [K]']>tmin)*(Cool['# log Teff [K]']<tmax)][::len(Cool)//50]
        #Cool.sort('Log(edad/Myr)')
        mass_array  = np.concatenate((mass_array, Cool['mass [Msun]'] ))
        logg        = np.concatenate(( logg, np.array(Cool['log g [cm/s^2]']) ))
        age         = np.concatenate(( age, Cool['total age [Gyr]'] * 1e9 ))
        age_cool    = np.concatenate(( age_cool, Cool['cooling age [Gyr]'] * 1e9 ))
        logteff     = np.concatenate(( logteff, np.array(Cool['# log Teff [K]']) ))
        Mbol        = np.concatenate(( Mbol, 4.75 - 2.5 * np.array(Cool['log L/Lsun']) ))

    return mass_array, logg, age, age_cool, logteff, Mbol


def slope(a,b):
        k1 = (a[1:-1]-a[:-2])/(b[1:-1]-b[:-2])
        k2 = (a[2:]-a[1:-1])/(b[2:]-b[1:-1])
        return k1 + (b[1:-1]-b[:-2])*(k1-k2)/(b[:-2]-b[2:])

def plot_slope(x,density,label,fmt=''):
    plt.plot(x[1:-1], density/density[~np.isnan(density)].mean(),\
                 fmt,label=label)
    return None

def open_22Ne_track(model,spec_type):
    #M 918,3304.8 V 5814 H 5936.4 V 3304.8 Z C
    #M 918,795.602 V 3304.8 H 5936.4 V 795.602 Z O   
    Ne_dtdlogL_T = Table.read(model+'.txt',format='csv',data_start=1)
    if '_O' in model:
        Ne_logLlogT = Table.read('22Ne_logLlogT_O.txt',format='csv',data_start=1)
        logteff_temp = (4.5-(np.cumsum(Ne_logLlogT['x'])-918)/(5936.4-918)*(4.5-3.4))
        Mbol_temp = 4.75-2.5*(-5.5+(np.cumsum(Ne_logLlogT['y'])-795.602)/(3304.8-795.602)*(5.5-1))
        logteff = np.log10((30-(np.cumsum(Ne_dtdlogL_T['x'])-918)/(5936.4-918)*(30-3))*1000)
        dtdMbol = 10**(-1.5+(np.cumsum(Ne_dtdlogL_T['y'])-795.602)/(3304.8-795.602)*(2+1.5))/2.5*1e9
    else:
        Ne_logLlogT = Table.read('22Ne_logLlogT.txt',format='csv',data_start=1)
        logteff_temp = (4.5-(np.cumsum(Ne_logLlogT['x'])-918)/(5936.4-918)*(4.5-3.4))
        Mbol_temp = 4.75-2.5*(-5.5+(np.cumsum(Ne_logLlogT['y'])-3304.8)/(5814-3304.8)*(5.5-1))
        logteff = np.log10((30-(np.cumsum(Ne_dtdlogL_T['x'])-918)/(5936.4-918)*(30-3))*1000)
        dtdMbol = 10**(-1.5+(np.cumsum(Ne_dtdlogL_T['y'])-3304.8)/(5814-3304.8)*(2+1.5))/2.5*1e9
    
    Mbol_logteff = interp1d(logteff_temp, Mbol_temp,bounds_error=False,fill_value='extrapolate')
    Mbol = Mbol_logteff(logteff)
    bp_rp_track = eval(spec_type+'_CO')['grid_bp_rp_func'](logteff,8.73253867 )
    G_track = eval(spec_type+'_CO')['grid_G_Mbol_func'](logteff,8.73253867 ) + Mbol
    t_Mbol = dtdMbol[1:-1]
    G_bprp = slope(G_track, bp_rp_track)
    Mbol_G = slope(Mbol, G_track)
    t_bprp = G_bprp * Mbol_G * t_Mbol
    t_G = t_Mbol * Mbol_G
    return bp_rp_track, G_track, t_bprp, t_G, t_Mbol, Mbol_G, G_bprp, np.ones(len(Mbol))*1.05, np.ones(len(Mbol))*8.73, None, None, logteff, Mbol, None

def inspect_a_track(spec_type,model,mass,IFMR,pl_type='bprp',pl=True):
    ## Open a Track
    if '22Ne' in model:
        bp_rp_track, G_track, t_bprp, t_G, t_Mbol, Mbol_G, G_bprp, \
        mass_array, logg, age, age_cool, logteff, Mbol, Mbol_logteff = open_22Ne_track(model,spec_type)
    else:
        mass_array, logg, age, age_cool, logteff, Mbol = \
                            open_a_track(spec_type,model,mass,IFMR)
        if 'Phase_Sep' in model:
            G_track = eval(spec_type+'_CO')['grid_G_Mbol_func'](logteff,logg) + Mbol
            bp_rp_track = eval(spec_type+'_CO')['grid_bp_rp_func'](logteff,logg)
        else: 
            G_track = eval(spec_type+'_'+model)['grid_G_Mbol_func'](logteff,logg) + Mbol
            bp_rp_track = eval(spec_type+'_'+model)['grid_bp_rp_func'](logteff,logg)
    
        t_bprp = slope(age_cool, bp_rp_track)
        t_Mbol = slope(age_cool, Mbol)
        t_G = slope(age_cool, G_track)
        G_bprp = slope(G_track, bp_rp_track)
        Mbol_G = slope(Mbol, G_track)
        Mbol_logteff = slope(Mbol, logteff)
    
    #def MF(mass_ini):
    #    return (mass_ini>0.58)*mass_ini**(-2.3)    
    if pl_type=='bprp':
        x = bp_rp_track
    if pl_type=='G':
        x = G_track
    if pl_type is not None:
        plt.plot(x[1:-1], t_bprp/10/1e9,\
                 label='Final: dt / d(bp-rp)',lw=10)
        plt.plot(x[1:-1], t_Mbol/1e9,\
                 label='dt / dMbol')
        #plot_slope(x,t_bprp,'Final: dt / d(bp-rp)')
        #plot_slope(x,t_Mbol,'dt / dMbol','.')
        plot_slope(x,Mbol_G,'dMbol / dG')
        plot_slope(x,G_bprp,'dG / d(bp-rp), slope on HR diagram')
    
        plt.title(spec_type+' '+model+' '+mass)
        plt.ylabel('[arbitrary unit]')
        if pl_type=='bprp':
            plt.xlabel('bp - rp')
        if pl_type=='G':
            plt.xlabel('G')
        plt.legend()
    return mass_array, logg, age, age_cool, logteff, Mbol, G_track, bp_rp_track, t_bprp, t_Mbol, t_G, G_bprp, Mbol_G, Mbol_logteff
