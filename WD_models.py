'''
This package will read different cooling models and interpolate the conversion functions
between HR diagram, Teff, Mbol, etc. The functions are stored in dictionaries for each model.
See the main function and the lines after its definition.
This package also contains the functions to read a single cooling track.
'''


import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii, votable
from astropy.table import Table, vstack, hstack
import os, sys
from astropy.io import fits
from scipy.interpolate import interp1d, interp2d, CloughTocher2DInterpolator, griddata, LinearNDInterpolator
from scipy.signal import fftconvolve

IFMR_new = interp1d((0.50, 0.60, 0.8, 0.95, 1.38),(0.8,2.75,3.75,6,10),\
                          fill_value = 0, bounds_error=False) # mass_WD, mass_ini
IFMR_new = interp1d((0.50, 0.55, 0.65,0.75, 0.85, 1.0, 1.25,1.35),(0.95,1,2,3,3.5,5,8,9),\
                          fill_value = 0, bounds_error=False) # mass_WD, mass_ini

IFMR_old = interp1d((0.64, 0.67, 0.8, 0.95, 1.38),(0.8,2.75,3.75,6,10),\
                          fill_value = 0, bounds_error=False) # mass_WD, mass_ini, for old star (>critical_age), we assume a 
t_index = -3

def MF(mass_ini):
    return (mass_ini>0.8)*mass_ini**(-2.3)

#----------------------------------------------------------------------------------------------------   


#----------------------------------------------------------------------------------------------------   

def interpolate_2d(x,y,para,method):
    if method == 'linear':
        interpolator = LinearNDInterpolator
    elif method == 'cubic':
        interpolator = CloughTocher2DInterpolator
    return interpolator((x,y), para, rescale=True)


def plot_lowmass_atm(spec_type='DB',color='G'):
    '''
    This function plots the tracks on logg vs. Teff diagram color coded by G or BP-RP.
    '''
    for mass in ['0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0','1.2']:
    #for mass in ['0.4']:
        Cool = Table.read('models/Fontaine_Gaia_atm_grid/Table_Mass_'+mass+'_'+spec_type,format='ascii')
        # every mass        
        if color == 'G':
            plt.scatter(np.log10(Cool['Teff']),Cool['logg'],c=Cool['G/R'],s=1,alpha=1,vmin=8,vmax=16)
        if color == 'G_Mbol':
            plt.scatter(np.log10(Cool['Teff']),Cool['logg'],c=Cool['G/R']-Cool['Mbol'],s=1,\
                        alpha=1,vmin=0,vmax=6)
        if color == 'bp_rp':
            plt.scatter(np.log10(Cool['Teff']),Cool['logg'],c=Cool['G_BP/R']-Cool['G_RP/R'],s=1,\
                        alpha=1,vmin=-0.8,vmax=1.25)
    Cool = Table.read('Fontaine_Gaia_atm_grid/Table_'+spec_type,format='ascii') # the whole grid
    if color == 'G':
        plt.scatter(np.log10(Cool['Teff']),Cool['logg'],c=Cool['G/R'],s=1,alpha=1,vmin=8,vmax=16)
    if color == 'G_Mbol':
        plt.scatter(np.log10(Cool['Teff']),Cool['logg'],c=Cool['G/R']-Cool['Mbol'],s=1,\
                    alpha=1,vmin=0,vmax=6)
    if color == 'bp_rp':
        plt.scatter(np.log10(Cool['Teff']),Cool['logg'],c=Cool['G_BP/R']-Cool['G_RP/R'],s=1,\
                    alpha=1,vmin=-0.8,vmax=1.25)
    return None


def interp_atm(spec_type='DB',color='G',xy=(4000,40000,100,7.0,9.5,0.01)):
    '''
    This function generates from the atmosphere model the function (logTeff, logg) --> G, BP-RP, or G-Mbol 
    
    arguments:
    color: the target variable of the interpolation function. 'G', 'BP-RP', or 'G-Mbol'.
    '''
    logTeff = np.zeros(0)
    logg = np.zeros(0)
    age = np.zeros(0)
    mass_array = np.zeros(0)
    G = np.zeros(0)
    bp_rp = np.zeros(0)
    Mbol = np.zeros(0)
    for mass in ['0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0','1.2']:
        Cool = Table.read('models/Fontaine_Gaia_atm_grid/Table_Mass_'+mass+'_'+spec_type,format='ascii')
        # every mass
        selected = Cool['Teff']>3500
        Cool = Cool[selected]
        
        bp_rp = np.concatenate((bp_rp,Cool['G_BP/R']-Cool['G_RP/R']))
        G = np.concatenate((G,Cool['G/R']))
        Mbol = np.concatenate((Mbol,Cool['Mbol']))
        logTeff = np.concatenate((logTeff,np.log10(Cool['Teff'])))
        logg = np.concatenate((logg,Cool['logg']))
        age = np.concatenate((age,Cool['Age']))
    Cool = Table.read('models/Fontaine_Gaia_atm_grid/Table_'+spec_type,format='ascii') # the whole grid
    selected = Cool['Teff']>3500
    Cool = Cool[selected]
    bp_rp = np.concatenate((bp_rp,Cool['G_BP/R']-Cool['G_RP/R']))
    G = np.concatenate((G,Cool['G/R']))
    Mbol = np.concatenate((Mbol,Cool['Mbol']))
    logTeff = np.concatenate((logTeff,np.log10(Cool['Teff'])))
    logg = np.concatenate((logg,Cool['logg']))
    age = np.concatenate((age,Cool['Age']))        
    
    grid_x, grid_y = np.mgrid[xy[0]:xy[1]:xy[2], xy[3]:xy[4]:xy[5]]
    def interp(x,y,para):
        grid_z = griddata(np.array((x,y)).T, para, (grid_x, grid_y), method=interp_type_atm)
        grid_z_func = interpolate_2d(x,y, para, interp_type_atm)
        return grid_z, grid_z_func
    
    if color == 'G':
        return interp(logTeff,logg,G)
    if color == 'bp_rp':
        return interp(logTeff,logg,bp_rp)
    if color == 'G_Mbol':
        return interp(logTeff,logg,G-Mbol)
    
    
def interp_compare(spec_type = 'DA',color = 'bp_rp',pl=False):
    '''
    plot the cooling tracks and the results of atmosphere interpolation on the
    logg vs. Teff diagram color coded by G or BP-RP
    '''
    interp_result, interp_func = interp_atm(spec_type=spec_type,color=color,\
                                            xy=(tmin,tmax,dt,loggmin,loggmax,dlogg))
    if pl==True:
        plt.figure(figsize=(12,1))
        plot_lowmass_atm(spec_type,color)
        #plot_lowmass(spec_type,'scatter',0,color)
        plt.xlim(tmin,tmax)
        plt.ylim(loggmin,loggmax)
        plt.colorbar()
        plt.show()
    
        plt.figure(figsize=(12,1))
        plt.imshow(interp_result.T, extent=(tmin,tmax,loggmin,loggmax), origin='lower',aspect='auto')
        plt.colorbar()
        plt.show()
    return interp_result, interp_func


def HR_to_para(para,xy,bp_rp_fontaine,G_fontaine,age,pl=False,title=' '):
    '''
    Interpolate the function of (BR-RP, G) --> para.
    (the argument 'age' is useless.)
    '''
    grid_x, grid_y = np.mgrid[xy[0]:xy[1]:xy[2], xy[3]:xy[4]:xy[5]]
    grid_x *= interp_bprp_factor
    selected = ~np.isnan(bp_rp_fontaine+G_fontaine+age+para)*(G_fontaine<16)*(G_fontaine>8)
    grid_z = griddata(np.array((bp_rp_fontaine[selected]*interp_bprp_factor, G_fontaine[selected])).T, \
                        para[selected], (grid_x, grid_y), method=interp_type)
    grid_z_func = interpolate_2d(bp_rp_fontaine[selected], G_fontaine[selected], \
                                 para[selected], interp_type)

    if pl==True:
        plt.figure(figsize=(12,3))
        plt.subplot(1,2,1)
        plt.title(title)
        plt.scatter(bp_rp_fontaine,G_fontaine,c=para,s=1,\
                    vmin=np.percentile(para[selected],10),vmax=np.percentile(para[selected],70))
        plt.xlim(xy[0],xy[1])
        plt.ylim(xy[3],xy[4])
        plt.colorbar()
        
        plt.subplot(1,2,2)
        plt.title(title)
        plt.imshow(grid_z.T, extent=(xy[0],xy[1],xy[3],xy[4]), origin='lower',aspect='auto',\
                  vmin=np.percentile(para[selected],10),vmax=np.percentile(para[selected],90))
        plt.colorbar()
        plt.show()
    return grid_z, grid_z_func


def interp_xy_z(para,xy,x,y,xfactor=1,pl=False,title=''):
    '''
    Interpolate the function (x,y) --> z from a series of x, y, and z values. 
    Similar to HR_to_para, but the x and y can be any variable.
    
    arguments:
    xy: the grid information. e.g. xy=(4000,40000,100,7.0,9.5,0.01). see np.mgrid for more information.
    '''
    grid_x, grid_y = np.mgrid[xy[0]:xy[1]:xy[2], xy[3]:xy[4]:xy[5]]
    grid_x *= xfactor
    selected = ~np.isnan(x+y+para)
    grid_z = griddata(np.array((x[selected]*xfactor, y[selected])).T, \
                        para[selected], (grid_x, grid_y), method=interp_type)
    grid_z_func = interpolate_2d(x[selected], y[selected], \
                                 para[selected], interp_type)
    if pl==True:
        plt.figure(figsize=(12,3))
        plt.subplot(1,2,1)
        plt.title(title)
        plt.scatter(x,y,c=para,s=1,\
                    vmin=np.percentile(para[selected],10),vmax=np.percentile(para[selected],70))
        plt.xlim(xy[0],xy[1])
        plt.ylim(xy[3],xy[4])
        plt.colorbar()
        
        plt.subplot(1,2,2)
        plt.title(title)
        plt.imshow(grid_z.T, extent=(xy[0],xy[1],xy[3],xy[4]), origin='lower',aspect='auto',\
                  vmin=np.percentile(para[selected],10),vmax=np.percentile(para[selected],90))
        plt.colorbar()
        plt.show()
    return grid_z, grid_z_func


def m_logage_to_HR(mass,logage,bp_rp_fontaine,G_fontaine):
    '''
    get the interpolated function (mass, logage) --> (BP_RP, G)
    '''
    selected = ~np.isnan(bp_rp_fontaine+G_fontaine+logage)*(G_fontaine<16)*(G_fontaine>8)
    grid_bprp_func = interpolate_2d( mass[selected], logage[selected], \
                                    bp_rp_fontaine[selected], interp_type)
    grid_G_func = interpolate_2d(mass[selected], logage[selected], \
                                 G_fontaine[selected], interp_type)
    return grid_bprp_func, grid_G_func


def open_evolution_tracks(model, spec_type, IFMR, logg_func=None):
    '''
    read the cooling models and store the following information of different cooling tracks together in one numpy array:
    mass, logg, age, age_for_density, logteff, Mbol
    '''
    logg = np.zeros(0); age = np.zeros(0); age_for_density = np.zeros(0)
    logteff = np.zeros(0); mass_array = np.zeros(0); Mbol = np.zeros(0)
    CO = ['020','030','040','050','060','070','080','090','095','100','105','110','115','120','125','130']
    ONe = ['020','030','040','050','060','070','080','090','095','100','105']
    MESA = ['020','030','040','050','060','070','080','090','095']
    Phase_Sep = ['054','055','061','068','077','087','100','110','120']
    if model=='CO':
        mass_list=CO
    if model=='CO+ONe':
        mass_list=ONe
    if model=='PG+ONe' or model=='PG+MESA' or ('Phase_Sep' in model):
        mass_list=[]
    if model=='CO+MESA':
        mass_list=MESA
    if model=='Phase_Sep+ONe':
        Phase_Sep = ['054','055','061','068','077','087','100']
    if spec_type=='DB':
        spec_suffix = '0210'; spec_suffix2 = 'DB'; spec_suffix3 = 'He'
    else:
        spec_suffix = '0204'; spec_suffix2 = 'DA'; spec_suffix3 = 'H'
    for mass in mass_list:
        f = open('models/Fontaine_AllSequences/CO_'+mass+spec_suffix)
        text = f.read()
        example = "      1    57674.0025    8.36722799  7.160654E+08  4.000000E+05  4.042436E+33\n        7.959696E+00  2.425570E+01  7.231926E+00  0.0000000000  0.000000E+00\n        6.019629E+34 -4.010597E+00 -1.991404E+00 -3.055254E-01 -3.055254E-01"
        logg_temp = []; age_temp = []; age_for_density_temp = []; logteff_temp = []; Mbol_temp = []
        for line in range(len(text)//len(example)):
            logteff_temp.append( np.log10(float(text[line*len(example)+9:line*len(example)+21])))
            logg_temp.append( float(text[line*len(example)+22:line*len(example)+35]))
            age_temp.append( float(text[line*len(example)+48:line*len(example)+63]) +\
                            (IFMR(int(mass)/100))**(t_index)*10**10)
            age_for_density_temp.append( float(text[line*len(example)+48:line*len(example)+63]) )
            Mbol_temp.append( 4.75-2.5*np.log10(float(text[line*len(example)+64:line*len(example)+76])/\
                            (3.828*10**33)) )
        mass_array = np.concatenate((mass_array,np.ones(len(logg_temp))*int(mass)/100))
        logg = np.concatenate((logg,logg_temp))
        age = np.concatenate((age,age_temp))
        age_for_density = np.concatenate((age_for_density,age_for_density_temp))
        logteff = np.concatenate((logteff,logteff_temp))
        Mbol = np.concatenate((Mbol,Mbol_temp))
    def smooth(x,window_len=5,window='hanning'):
        w=eval('np.'+window+'(window_len)')
        y=np.convolve(w/w.sum(),x,mode='same')
        return x
    
    if model=='Phase_Sep' or model=='Phase_Sep+ONe':
        for mass in Phase_Sep:
            Cool = Table.read('models/BaSTI/'+'COOL'+mass+'BaSTIfinale'+spec_suffix2+'sep.sdss',
                              format='ascii') 
            Cool = Cool[(Cool['log(Teff)']>tmin)*(Cool['log(Teff)']<tmax)][::len(Cool)//100]
            #Cool.sort('Log(edad/Myr)')
            mass_array = np.concatenate((mass_array,np.ones(len(Cool))*int(mass)/100))
            Cool['Log(grav)'] = logg_func( np.array(Cool['log(Teff)']), np.ones(len(Cool))*int(mass)/100 )
            logg = np.concatenate(( logg,smooth(np.array(Cool['Log(grav)'])) ))
            age = np.concatenate(( age,10**smooth(np.array(Cool['log(t)'])) -\
                                  10**Cool['log(t)'][0] +\
                                 (IFMR(int(mass)/100))**(t_index)*10**10 ))
            age_for_density = np.concatenate(( age_for_density,\
                                  10**smooth(np.array(Cool['log(t)'])) -\
                                  10**Cool['log(t)'][0]  ))   ## this is only the WD age
            logteff = np.concatenate(( logteff,smooth(np.array(Cool['log(Teff)'])) ))
            Mbol = np.concatenate(( Mbol, 4.75-2.5*smooth(np.array(Cool['log(L/Lo)'])) ))
            
    if 'Phase_Sep_' in model:
        for time in ['200','300','400','500','600','700','800','900','1000',
                    '1250','1500','1750','2000','2250','2500','2750','3000',
                    '3250','3500','3750','4000','4250','4500','4750','5000',
                    '5250','5500','5750','6000','6250','6500','6750','7000',
                    '7250','7500','7750','8000','8250','8500','8750','9000',
                    '9500','10000','10500','11000','11500','12000','12500','13000','13500','14000']:
            if '4' in model:
                Cool = Table.read('models/BaSTI_z42aeo/WDz402y303aenot'+time+'.'+spec_suffix2+'sep.sdss',
                              format='ascii') 
            if '2' in model:
                Cool = Table.read('models/BaSTI_z22aeo/WDzsunysunaenot'+time+'.'+spec_suffix2+'sep.sdss',
                              format='ascii') 
            Cool = Cool[(Cool['log(Teff)']>tmin)*(Cool['log(Teff)']<tmax)][::len(Cool)//100]
            #Cool.sort('Log(edad/Myr)')
            mass_array = np.concatenate((mass_array,np.array(Cool['Mwd']) ))
            Cool['Log(grav)'] = logg_func( np.array(Cool['log(Teff)']), np.array(Cool['Mwd']) )
            logg = np.concatenate(( logg,smooth(np.array(Cool['Log(grav)'])) ))
            age = np.concatenate(( age,(np.ones(len(Cool))*int(time)*1e6) +\
                                 IFMR(np.array(Cool['Mwd']))**(t_index)*1e10 ))
            age_for_density = np.concatenate(( age_for_density,\
                                  (np.ones(len(Cool))*int(time)*1e6)   ))   ## this is only the WD age
            logteff = np.concatenate(( logteff,smooth(np.array(Cool['log(Teff)'])) ))
            Mbol = np.concatenate(( Mbol, 4.75-2.5*smooth(np.array(Cool['log(L/Lo)'])) ))
    
    if model=='PG+ONe':
        for mass in ['0514','0530','0542','0565','0584','0609','0664','0741','0869']:
            Cool = Table.read('models/tracksPG-DB/db-pg-'+mass+'.trk.t0.11',format='ascii',comment='#')
            Cool = Cool[(Cool['Log(Teff)']>tmin)*(Cool['Log(Teff)']<tmax)*(Cool['age[Myr]']>0)]
            mass_array = np.concatenate((mass_array,np.ones(len(Cool))*int(mass)/1000))
            logg = np.concatenate(( logg,smooth(np.array(Cool['Log(grav)'])) ))
            age = np.concatenate(( age,np.array(Cool['age[Myr]'])*10**6 +\
                                  (IFMR(int(mass)/1000))**(t_index)*10**10 ))
            age_for_density = np.concatenate(( age_for_density,\
                                              np.array(Cool['age[Myr]'])*10**6 ))
            logteff = np.concatenate(( logteff,smooth(np.array(Cool['Log(Teff)'])) ))
            Mbol = np.concatenate(( Mbol, 4.75-2.5*smooth(np.array(Cool['Log(L)'])) ))
            
    # incorperate massive ONe model
    if model=='CO+ONe' or model=='PG+ONe' or model=='Phase_Sep+ONe' or ('Phase_Sep_' in model): 
        for mass in ['110','116','122','129']:
            Cool = Table.read('models/ONeWDs/'+mass+'_'+spec_suffix2+'.trk',format='ascii') 
            Cool = Cool[(Cool['LOG(TEFF)']>tmin)*(Cool['LOG(TEFF)']<tmax)][::10]
            Cool.sort('Log(edad/Myr)')
            mass_array = np.concatenate((mass_array,np.ones(len(Cool))*int(mass)/100))
            logg = np.concatenate(( logg,smooth(np.array(Cool['Log(grav)'])) ))
            age = np.concatenate(( age,10**smooth(np.array(Cool['Log(edad/Myr)']))*10**6 -\
                                  10**Cool['Log(edad/Myr)'][0]*10**6+\
                                (IFMR(int(mass)/100))**(t_index)*10**10 ))
            age_for_density = np.concatenate(( age_for_density,\
                                              10**smooth(np.array(Cool['Log(edad/Myr)']))*10**6-\
                                              10**Cool['Log(edad/Myr)'][0]*10**6))
            logteff = np.concatenate(( logteff,smooth(np.array(Cool['LOG(TEFF)'])) ))
            Mbol = np.concatenate(( Mbol, 4.75-2.5*smooth(np.array(Cool['LOG(L)'])) ))
            
    # incorperate massive MESA model
    if model=='CO+MESA' or model=='PG+MESA': 
        if spec_suffix3 == 'He':
            mesa_masslist = ['1.0124','1.019','1.0241','1.0358','1.0645','1.0877','1.1102','1.1254','1.1313','1.1322',\
                     '1.1466','1.151','1.2163','1.22','1.2671','1.3075']
        else:
            mesa_masslist = ['1.0124','1.019','1.0241','1.0358','1.0645','1.0877','1.1102','1.125','1.1309','1.1322',\
                     '1.1466','1.151','1.2163','1.22','1.2671','1.3075']
        for mass in mesa_masslist:
            Cool = Table.read('models/MESA_model/'+spec_suffix3+'_atm-M'+mass+'.dat',format='csv',header_start=1,data_start=2) 
            Cool = Cool[(Cool['# log Teff [K]']>tmin)*(Cool['# log Teff [K]']<tmax)][::1]
            #Cool.sort('Log(edad/Myr)')
            mass_array = np.concatenate((mass_array,Cool['mass [Msun]']))
            logg = np.concatenate(( logg,np.array(Cool['log g [cm/s^2]']) ))
            age = np.concatenate(( age,Cool['total age [Gyr]'] * 1e9 ))
            age_for_density = np.concatenate(( age_for_density,\
                                              Cool['cooling age [Gyr]'] * 1e9))
            logteff = np.concatenate(( logteff,np.array(Cool['# log Teff [K]']) ))
            Mbol = np.concatenate(( Mbol, 4.75-2.5*np.array(Cool['log L/Lsun']) ))
    
    select = ~np.isnan(mass_array + logg + age + age_for_density + logteff + Mbol) * (np.log10(age_for_density)>-10)
    return mass_array[select], logg[select], age[select], age_for_density[select], logteff[select], Mbol[select]


def plot_G_bprp_density(G, bp_rp, density, mass_array,age):
    '''
    plot the number-density on the HR diagram predicted by the cooling model, assuming constant star formation rate.
    '''
    plt.figure(figsize=(18,5))
    plt.subplot(1,3,1)
    plt.scatter(bp_rp_fontaine,G_fontaine,c=mass_array,s=1)
    plt.colorbar();plt.ylabel('G');plt.title('mass [Msol]');plt.ylim(8,16)
    plt.subplot(1,3,2)
    plt.scatter(bp_rp_fontaine,G_fontaine,c=np.log10(age),s=1)
    plt.colorbar();plt.ylabel('G');plt.title('log age [yr]');plt.ylim(8,16)
    plt.subplot(1,3,3)
    plt.scatter(bp_rp_fontaine,G_fontaine,c=density,s=1,vmin=0,vmax=1)
        #np.percentile(density[density>0],90))
    plt.colorbar();plt.ylabel('G');plt.title('density in color [yr/mag]');plt.ylim(8,16)
    plt.show()
    return None


def plot_contour_age_mass(grid_logage,grid_mass,x_shift=0,y_shift=0,age_levels=[1,2,3,4,5,6,7,8,9],age_labels=[5],\
                          mass_levels=[0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3],mass_labels=[0.8]):
    '''
    plot the contour of WD age and mass on the HR diagram.
    '''
    CS = plt.contour(10**grid_logage.T/10**9,levels=age_levels,linestyles='dashed',cmap='jet',\
                extent=(xy[0]+x_shift,xy[1]+x_shift,xy[3]+y_shift,xy[4]+y_shift),\
                origin='lower',aspect='auto')
    plt.clabel(CS, age_labels, fontsize=9, inline=1)
    CS = plt.contour(grid_mass.T,levels=mass_levels,\
                extent=(xy[0]+x_shift,xy[1]+x_shift,xy[3]+y_shift,xy[4]+y_shift),\
                origin='lower',aspect='auto')
    plt.clabel(CS, mass_labels, fontsize=9, inline=1)
    return None


def plot_contours(grid_object,levels,marks,x_shift,stars='on',linestyle='dashed',**kwarg):
    CS = plt.contour(grid_object.T,levels=levels,linestyles=linestyle,cmap='jet',\
                extent=(xy[0],xy[1],xy[3],xy[4]),\
                origin='lower',aspect='auto',alpha=1,**kwarg)
    plt.clabel(CS, marks, fontsize=9, inline=1)
    
    if stars=='on':
        #table = np.load('WD_planet.npy')[0]['WD_planet']
        #high_snr = (table['parallax_over_error']>10) * (1/table['parallax']<distance_range/1000)
        table = np.load('WD_selected.npy')[0]['WD']
        high_snr = (table['parallax_over_error']>10) * (1/table['parallax']<distance_range/1000)
    
        x = table['bp_rp'][high_snr]+x_shift
        y = (table['phot_g_mean_mag']+5*np.log10(table['parallax']/1000)+5)[high_snr]
        plt.plot(x,y,'.b',markersize=1)

    plt.ylim(16,8)
    plt.xlim(-0.6,2.5)
    plt.grid()
    return None


def final_plot(grid_para,grid_logage,grid_mass,horizontal_shift,vertical_shift,bright_end=85,\
               stars='on',distance_range=100,\
              age_levels=[1,2,3,4,5,6,7,8,9],age_labels=[5],\
               mass_levels=[0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3],mass_labels=[0.8],**kwarg):
    # Contour of Age (from ZAMS) and Mass
    plot_contour_age_mass(grid_logage,grid_mass,age_levels=age_levels,age_labels=age_labels,\
                          mass_levels=mass_levels,\
                         mass_labels=mass_labels)
    # Density Hess
    selected = grid_para>0
    vmin = 0;#np.percentile(grid_para[selected],5)
    vmax = np.percentile(grid_para[selected],bright_end)
    plt.imshow(grid_para.T, extent=(xy[0],xy[1],xy[3],xy[4]), origin='lower',aspect='auto',\
              vmin=vmin,vmax=vmax,**kwarg)
    
    #table = np.load('WD_planet.npy')[0]['WD_planet']
    #high_snr = (table['parallax_over_error']>10) * (1/table['parallax']<distance_range/1000)
    table = np.load('WD_selected.npy')[0]['WD']
    high_snr = (table['parallax_over_error']>10) * (1/table['parallax']<distance_range/1000)
    
    # Plot Shifted Comparison
        # Gaia WDs
    if stars=='on':
        x = table['bp_rp'][high_snr] + horizontal_shift
        y = (table['phot_g_mean_mag'] + 5*np.log10(table['parallax']/1000)+5)[high_snr] + vertical_shift
        plt.plot(x,y,'.b',markersize=1)
        # Shifted Contour of Age and Mass
    plot_contour_age_mass(grid_logage,grid_mass,age_levels=age_levels,age_labels=age_labels,\
                          mass_levels=mass_levels,\
                         mass_labels=mass_labels)
    
    plt.ylim(16,8)
    plt.xlim(-0.6,2.5)
    plt.grid()


#----------------------------------------------------------------------------------------------------   


#----------------------------------------------------------------------------------------------------   

def main(spec_type, model,IFMR, logg_func=None):
    # Make Atmosphere Grid/Function: logTeff, logg --> bp-rp,  G-Mbol
    grid_G_Mbol, grid_G_Mbol_func = interp_compare(spec_type,'G_Mbol',pl=pl[0])
    grid_bp_rp, grid_bp_rp_func = interp_compare(spec_type,'bp_rp',pl=pl[0])
    
    
    # Open Evolution Tracks
    mass_array, logg, age, age_for_density, logteff, Mbol = open_evolution_tracks(model, spec_type, IFMR, logg_func)
    

    # Get Colour/Magnitude for Evolution Tracks
    G_fontaine = grid_G_Mbol_func(logteff,logg) + Mbol
    bp_rp_fontaine = grid_bp_rp_func(logteff,logg)
    
    
    # Calculate Density on HR Diagram
    k1 = (age_for_density[1:-1]-age_for_density[:-2])/(bp_rp_fontaine[1:-1]-bp_rp_fontaine[:-2])
    k2 = (age_for_density[2:]-age_for_density[1:-1])/(bp_rp_fontaine[2:]-bp_rp_fontaine[1:-1])
    k = k1 + (bp_rp_fontaine[1:-1]-bp_rp_fontaine[:-2])*(k1-k2)/(bp_rp_fontaine[:-2]-bp_rp_fontaine[2:])
    density = np.concatenate((np.array([1]), k , np.array([1])))
        
    
    # Get Parameters on HR Diagram
    grid_logage, grid_logage_func = HR_to_para(np.log10(age),xy,bp_rp_fontaine,G_fontaine,age,\
                                               pl[1],'log age [yr]')
    grid_mass, grid_mass_func = HR_to_para(mass_array,xy,bp_rp_fontaine,G_fontaine,age,\
                                           pl[1],'mass [Msol]')
    row,col = grid_mass.shape
    grid_mass_density = np.concatenate((np.zeros((row,1)),\
                                        grid_mass[:,2:] - grid_mass[:,:-2] ,\
                                        np.zeros((row,1)) ), axis=1)
    grid_density, grid_density_func = HR_to_para(density,xy,bp_rp_fontaine,G_fontaine,age,\
                                                 pl[1],'density')
    grid_MF = MF(IFMR(grid_mass))
    grid_MF_func = lambda x,y: MF(IFMR(grid_mass_func(x,y)))
    grid_teff, grid_teff_func = HR_to_para(10**logteff,xy,bp_rp_fontaine,G_fontaine,age,False)
    grid_logage_for_density, grid_logage_for_density_func = HR_to_para(np.log10(age_for_density),\
                                          xy,bp_rp_fontaine,G_fontaine,age,False)
    grid_Mbol, grid_Mbol_func = HR_to_para(Mbol,xy,bp_rp_fontaine,G_fontaine,age,False)
    
    grid_bprp_func, grid_G_func = \
        m_logage_to_HR(mass_array,np.log10(age_for_density),bp_rp_fontaine,G_fontaine)
    
    
    
    # A Plot of Comparison of Data and Grid for Parameters
    if pl[2]==True:
        plot_G_bprp_density(G_fontaine, bp_rp_fontaine, density, mass_array, age)
    
    
    # The Final Plot
    if pl[3]==True:
        plt.figure(figsize=(12,6))
        final_plot(grid_density * grid_MF,grid_logage,grid_mass,horizontal_shift,vertical_shift,\
                   distance_range=distance_range)
        plt.title(spec+' '+model)
        plt.show()
    
    # Return a dictionary containing all the cooling track data points, interpolation functions and interpolation grids 
    return {'grid_G_Mbol':grid_G_Mbol, 'grid_G_Mbol_func':grid_G_Mbol_func,\
            'grid_bp_rp':grid_bp_rp, 'grid_bp_rp_func':grid_bp_rp_func,\
            'mass_array':mass_array, 'logg':logg, 'age':age, 'age_for_density':age_for_density,\
            'logteff':logteff, 'Mbol':Mbol, 'density':density,\
            'G_fontaine':G_fontaine, 'bp_rp_fontaine':bp_rp_fontaine,\
            'grid_logage':grid_logage, 'grid_logage_func':grid_logage_func,\
            'grid_mass':grid_mass, 'grid_mass_func':grid_mass_func,\
            'grid_density':grid_density, 'grid_density_func':grid_density_func,\
            'grid_MF':grid_MF, 'grid_MF_func':grid_MF_func,\
            'grid_teff':grid_teff, 'grid_teff_func':grid_teff_func,\
            'grid_logage_for_density':grid_logage_for_density, 'grid_logage_for_density_func':grid_logage_for_density,\
            'grid_Mbol':grid_Mbol, 'grid_Mbol_func':grid_Mbol_func,\
            'grid_mass_density':grid_mass_density,\
            'grid_bprp_func':grid_bprp_func, 'grid_G_func':grid_G_func}


tmin = 3.5; tmax = 5.1; dt = 0.01
loggmin = 6.5; loggmax = 9.6; dlogg = 0.01
xy = (-0.6, 1.5, 0.002, 10, 15, 0.01) # bp_rp, G
pl = [0,0,0,0]
horizontal_shift = 1
vertical_shift = -2
distance_range = 200 # pc
interp_type = 'linear'
interp_type_atm = 'linear'
interp_bprp_factor = 5

# Fontaine et al. 2001 (CO), Camisassa et al. 2019 (ONe), PG, and Lauffer et al. 2019 (MESA) models
DA_thick_CO = main('DA_thick','CO',IFMR_new)
DA_thin_CO = main('DA_thin','CO',IFMR_new)
DB_CO= main('DB','CO',IFMR_new)
DA_thick_ONe = main('DA_thick','CO+ONe',IFMR_new)
DA_thin_ONe = main('DA_thin','CO+ONe',IFMR_new)
DB_ONe = main('DB','CO+ONe',IFMR_new)
DB_PGONe = main('DB','PG+ONe',IFMR_new)
DA_thick_MESA = main('DA_thick','CO+MESA',IFMR_new)
DB_MESA = main('DB','CO+MESA',IFMR_new)

# get BaSTI logg_func
_, logg_func_DA_thick_CO = interp_xy_z(DA_thick_CO['logg'], [2.8,5.2,0.02,0.38,1.35,0.01], DA_thick_CO['logteff'],
                                 DA_thick_CO['mass_array'],)
_, logg_func_DA_thin_CO = interp_xy_z(DA_thin_CO['logg'], [2.8,5.2,0.02,0.38,1.35,0.01], DA_thin_CO['logteff'],
                                 DA_thin_CO['mass_array'],)
_, logg_func_DB_CO = interp_xy_z(DB_CO['logg'], [2.8,5.2,0.02,0.38,1.35,0.01], DB_CO['logteff'],
                                 DB_CO['mass_array'],)

# Salaris et al. 2010 (Phase_Sep) BaSTI models. 4 and 2 are alpha-enhanced models.
DA_thick_Phase_Sep = main('DA_thick','Phase_Sep',IFMR_new, logg_func_DA_thick_CO)
DB_Phase_Sep = main('DB','Phase_Sep',IFMR_new, logg_func_DB_CO)

DA_thick_Phase_Sep_4 = main('DA_thick','Phase_Sep_4',IFMR_new, logg_func_DA_thick_CO)
DB_Phase_Sep_4 = main('DB','Phase_Sep_4',IFMR_new, logg_func_DB_CO)

DA_thick_Phase_Sep_2 = main('DA_thick','Phase_Sep_2',IFMR_new, logg_func_DA_thick_CO)
DB_Phase_Sep_2 = main('DB','Phase_Sep_2',IFMR_new, logg_func_DB_CO)

#DA_thick_CO_old = main('DA_thick','CO',IFMR_old)
#DA_thin_CO_old = main('DA_thin','CO',IFMR_old)
#DB_CO_old= main('DB','CO',IFMR_old)
#DA_thick_ONe_old = main('DA_thick','CO+ONe',IFMR_old)
#DA_thin_ONe_old = main('DA_thin','CO+ONe',IFMR_old)
#DB_ONe_old = main('DB','CO+ONe',IFMR_old)
#DB_PGONe_old = main('DB','PG+ONe',IFMR_old)





#----------------------------------------------------------------------------------------------------   


#----------------------------------------------------------------------------------------------------   

# Inspect One Evolutionary Track
def open_a_track(spec_type,model,mass,IFMR,logg_func=None):
    tmin = 2.96; tmax = 5; dt = 0.01
    loggmin = 6.5; loggmax = 9.5; dlogg = 0.01
    
    logg = np.zeros(0)
    age = np.zeros(0)
    age_for_density = np.zeros(0)
    logteff = np.zeros(0)
    mass_array = np.zeros(0)
    Mbol = np.zeros(0)

    if spec_type=='DB':
        spec_suffix = '0210'; spec_suffix2 = 'DB'; spec_suffix3 = 'He'
    else:
        spec_suffix = '0204'; spec_suffix2 = 'DA'; spec_suffix3 = 'H'
        
    if model=='CO':  
        CO_masslist = ['020','030','040','050','060','070','080','090','095','100','105','110','115','120','125','130']
        mass_diff = 1000; mass_temp=''
        for mass_CO in CO_masslist:
            if abs(float(mass)-float(mass_CO)) < mass_diff:
                mass_diff = abs(float(mass)-float(mass_CO))
                mass_temp = mass_CO
        mass = mass_temp
        f = open('models/Fontaine_AllSequences/CO_'+mass+spec_suffix)
        text = f.read()
        example = "      1    57674.0025    8.36722799  7.160654E+08  4.000000E+05  4.042436E+33\n        7.959696E+00  2.425570E+01  7.231926E+00  0.0000000000  0.000000E+00\n        6.019629E+34 -4.010597E+00 -1.991404E+00 -3.055254E-01 -3.055254E-01"
        logg_temp = []
        age_temp = []
        age_for_density_temp = []
        logteff_temp = []
        Mbol_temp = []
        for line in range(len(text)//len(example)):
            logteff_temp.append( np.log10(float(text[line*len(example)+9:line*len(example)+21])))
            logg_temp.append( float(text[line*len(example)+22:line*len(example)+35]))
            age_temp.append( float(text[line*len(example)+48:line*len(example)+63]) +\
                            (IFMR(int(mass)/100))**(t_index)*10**10)
            age_for_density_temp.append( float(text[line*len(example)+48:line*len(example)+63]) )
            Mbol_temp.append( 4.75-2.5*np.log10(float(text[line*len(example)+64:line*len(example)+76])/\
                            (3.828*10**33)) )
        mass_array = np.concatenate((mass_array,np.ones(len(logg_temp))*int(mass)/100))
        logg = np.concatenate((logg,logg_temp))
        age = np.concatenate((age,age_temp))
        age_for_density = np.concatenate((age_for_density,age_for_density_temp))
        logteff = np.concatenate((logteff,logteff_temp))
        Mbol = np.concatenate((Mbol,Mbol_temp))
        
    if model=='ONe':
        def smooth(x,window_len=10,window='hanning'):
            s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
            w=eval('np.'+window+'(window_len)')
            y=np.convolve(w/w.sum(),s,mode='same')
            return x
        if spec_type=='DB':
            spec_suffix2 = 'DB'
        else:
            spec_suffix2 = 'DA'
        Cool = Table.read('models/ONeWDs/'+mass+'_'+spec_suffix2+'.trk',format='ascii') 
        Cool = Cool[(Cool['LOG(TEFF)']>tmin)*(Cool['LOG(TEFF)']<tmax)][::len(Cool)//50]
        Cool.sort('Log(edad/Myr)')
        mass_array = np.concatenate((mass_array,np.ones(len(Cool))*int(mass)/100))
        logg = np.concatenate(( logg,smooth(np.array(Cool['Log(grav)'])) ))
        age = np.concatenate(( age,10**smooth(np.array(Cool['Log(edad/Myr)']))*10**6 -\
                              10**Cool['Log(edad/Myr)'][0]*10**6))
        age_for_density = np.concatenate(( age_for_density,\
                                          10**smooth(np.array(Cool['Log(edad/Myr)']))*10**6-\
                                          10**Cool['Log(edad/Myr)'][0]*10**6 -\
                                         (IFMR(int(mass)/100))**(t_index)*10**10))
        logteff = np.concatenate(( logteff,smooth(np.array(Cool['LOG(TEFF)'])) ))
        Mbol = np.concatenate(( Mbol, 4.75-2.5*smooth(np.array(Cool['LOG(L)'])) ))
        
    if model=='Phase_Sep':
        def smooth(x,window_len=10,window='hanning'):
            s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
            w=eval('np.'+window+'(window_len)')
            y=np.convolve(w/w.sum(),s,mode='same')
            return x
        if spec_type=='DB':
            spec_suffix2 = 'DB'
        else:
            spec_suffix2 = 'DA'
        Cool = Table.read('models/BaSTI/'+'COOL'+mass+'BaSTIfinale'+spec_suffix2+'sep.sdss',
                          format='ascii') 
        Cool = Cool[(Cool['log(Teff)']>tmin)*(Cool['log(Teff)']<tmax)][::len(Cool)//100]
        #Cool.sort('Log(edad/Myr)')
        mass_array = np.concatenate((mass_array,np.ones(len(Cool))*int(mass)/100))
        Cool['Log(grav)'] = logg_func( np.array(Cool['log(Teff)']), np.ones(len(Cool))*int(mass)/100 )
        logg = np.concatenate(( logg,smooth(np.array(Cool['Log(grav)'])) ))
        age = np.concatenate(( age,10**smooth(np.array(Cool['log(t)'])) -\
                              10**Cool['log(t)'][0] +\
                             (IFMR(int(mass)/100))**(t_index)*10**10 ))
        age_for_density = np.concatenate(( age_for_density,\
                              10**smooth(np.array(Cool['log(t)'])) -\
                              10**Cool['log(t)'][0]  ))   ## this is only the WD age
        logteff = np.concatenate(( logteff,smooth(np.array(Cool['log(Teff)'])) ))
        Mbol = np.concatenate(( Mbol, 4.75-2.5*smooth(np.array(Cool['log(L/Lo)'])) ))
    
    if model=='PGONe':
        PG_masslist = ['0514','0530','0542','0565','0584','0609','0664','0741','0869']
        mass_diff = 10000; mass_temp=''
        for mass_PG in PG_masslist:
            if abs(float(mass)-float(mass_PG)) < mass_diff:
                mass_diff = abs(float(mass)-float(mass_PG))
                mass_temp = mass_PG
        mass = mass_temp
        Cool = Table.read('models/tracksPG-DB/db-pg-'+mass+'.trk.t0.11',format='ascii',comment='#')
        Cool = Cool[(Cool['Log(Teff)']>tmin)*(Cool['Log(Teff)']<tmax)*(Cool['age[Myr]']>0)][::len(Cool)//200]
        mass_array = np.concatenate((mass_array,np.ones(len(Cool))*int(mass)/1000))
        logg = np.concatenate(( logg,(np.array(Cool['Log(grav)'])) ))
        age = np.concatenate(( age,np.array(Cool['age[Myr]'])*10**6 +\
                              (IFMR(int(mass)/1000))**(t_index)*10**10 ))
        age_for_density = np.concatenate(( age_for_density,\
                                          np.array(Cool['age[Myr]'])*10**6 ))
        logteff = np.concatenate(( logteff,(np.array(Cool['Log(Teff)'])) ))
        Mbol = np.concatenate(( Mbol, 4.75-2.5*(np.array(Cool['Log(L)'])) ))
        
    if model=='MESA':
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
        mass_array = np.concatenate((mass_array,Cool['mass [Msun]']))
        logg = np.concatenate(( logg,np.array(Cool['log g [cm/s^2]']) ))
        age = np.concatenate(( age,Cool['total age [Gyr]'] * 1e9 ))
        age_for_density = np.concatenate(( age_for_density,\
                                          Cool['cooling age [Gyr]'] * 1e9))
        logteff = np.concatenate(( logteff,np.array(Cool['# log Teff [K]']) ))
        Mbol = np.concatenate(( Mbol, 4.75-2.5*np.array(Cool['log L/Lsun']) ))

    return mass_array, logg, age, age_for_density, logteff, Mbol


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
        mass_array, logg, age, age_for_density, logteff, Mbol, Mbol_logteff = open_22Ne_track(model,spec_type)
    else:
        mass_array, logg, age, age_for_density, logteff, Mbol = \
                            open_a_track(spec_type,model,mass,IFMR)
        if 'Phase_Sep' in model:
            G_track = eval(spec_type+'_CO')['grid_G_Mbol_func'](logteff,logg) + Mbol
            bp_rp_track = eval(spec_type+'_CO')['grid_bp_rp_func'](logteff,logg)
        else: 
            G_track = eval(spec_type+'_'+model)['grid_G_Mbol_func'](logteff,logg) + Mbol
            bp_rp_track = eval(spec_type+'_'+model)['grid_bp_rp_func'](logteff,logg)
    
        t_bprp = slope(age_for_density, bp_rp_track)
        t_Mbol = slope(age_for_density, Mbol)
        t_G = slope(age_for_density, G_track)
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
    return mass_array, logg, age, age_for_density, logteff, Mbol, G_track, bp_rp_track, t_bprp, t_Mbol, t_G, G_bprp, Mbol_G, Mbol_logteff