# WD_models
I provide a python module for transformation between the *Gaia* H--R diagram and many white dwarf (WD) physical parameters (L, Teff, log *g*, etc), based on an interpolation of cooling tracks from various existing WD models.

This module is a python 3 script and will use functions from the following packages:
```
astropy, matplotlib, numpy, scipy
```

One of the main usage of this module is to convert H--R diagram coordinate into WD parameters. Below is an example:
it will first read the cooling tracks of hydrogen-atmosphere (DA) WDs from the "Fontaine2001" model for low-mass WDs (<0.5 Msun), from the "Althaus2010, Z=0.01" model for normal-mass WDs (0.5 to 1.0 Msun), and from the "ONe" model for high-mass WDs (>1.0 Msun); then, it will calculate the cooling age for two sample coordinates on the H--R diagram.
```
import WD_models

# loading a set of models
model = WD_models.load_model(low_mass_model='Fontaine2001',
                             normal_mass_model='Althaus2010_001',
                             high_mass_model='ONe',
                             spec_type='DA_thick',
                             )
                             
# calculate the cooling age (in Gyr) of two WDs with
# (BP-RP, G) = (0.25, 13) and (0.25, 14)
print(model['HR_to_age_cool']([0.25, 0.25], [13,14]))
```
The output is: `>> array([  1.27785237e+09,   2.70284467e+09])`

The function `load_model` reads a set of cooling tracks assigned by the user and returns many useful functions for mapping between Gaia photometry and WD parameters and grid-values for plotting the contour of WD parameters. For other mappings not included in this output, the user can generate the interpolated grid values and mapping function based on the cooling-track data points and atmosphere models that are also provided in the output. 
E.g., for the mapping (mass, logteff) --> cooling age,
```
model = WD_models.load_model('f', 'a001', 'o', 'DA_thick')
m_logteff_to_agecool = WD_models.interp_xy_z_func(
    model['mass_array'], model['logteff'], model['age_cool'], 'linear')
```
Note that the name of models can be abbreviated . The details are listed below:


low_mass_model:     String. Specifying the cooling model used for low-
                        mass WDs (<~0.5Msun). Its value should be one of the
                            following: 
            ''                              no low-mass model will be read
            'Fontaine2001' or 'f'           http://www.astro.umontreal.ca/~bergeron/CoolingModels/
        normal_mass_model:  String. Specifying the cooling model used for 
                            normal-mass WDs (about 0.5~1.0Msun). Its value 
                            should be one of the following:
            ''                              no normal-mass model will be read
            'Fontaine2001' or 'f'           http://www.astro.umontreal.ca/~bergeron/CoolingModels/
            'Althaus2010_001' or 'a001'     Z=0.01, only for DA, http://evolgroup.fcaglp.unlp.edu.ar/TRACKS/tracks_cocore.html
            'Althaus2010_0001' or 'a0001'   Z=0.001, only for DA, http://evolgroup.fcaglp.unlp.edu.ar/TRACKS/tracks_cocore.html
            'Camisassa2017' or 'c'          only for DB, http://evolgroup.fcaglp.unlp.edu.ar/TRACKS/tracks_DODB.html
            'BaSTI' or 'b'                  with phase separation, Salaris et al. 2010, http://basti.oa-teramo.inaf.it
            'BaSTI_nosep' or 'bn'           no phase separation, Salaris et al. 2010, http://basti.oa-teramo.inaf.it
            'PG'                            only for DB
        high_mass_model:    String. Specifying the cooling model used for 
                            high-mass WDs (>~1.0Msun). Should be one of the
                            following: 
            ''                              no high-mass model will be read
            'Fontaine2001' or 'f'           http://www.astro.umontreal.ca/~bergeron/CoolingModels/
            'ONe' or 'o'                    Camisassa et al. 2019, http://evolgroup.fcaglp.unlp.edu.ar/TRACKS/ultramassive.html
            'MESA' or 'm'                   Lauffer et al. 2019
            'BaSTI' or 'b'                  with phase separation, Salaris et al. 2010, http://basti.oa-teramo.inaf.it
            'BaSTI_nosep' or 'bn'           no phase separation, Salaris et al. 2010, http://basti.oa-teramo.inaf.it
        spec_type:          String. Specifying the atmosphere composition.
                            Its value should be one of the following:
            'DA_thick'                      thick hydrogen atmosphere
            'DA_thin'                       thin hydrogen atmosphere
            'DB'                            pure-helium atmosphere
        logg_func:          Function. 
            This is a function for (logteff, mass) --> logg. It is necessary 
            only for BaSTI models, because the BaSTI models do not directly 
            provide log g information.
        for_comparison:     Bool. 
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
       atm.      | 'grid_logteff_logg_to_G_Mbol' | 'logteff_logg_to_G_Mbol'
                 | 'grid_logteff_logg_to_bp_rp'  | 'logteff_logg_to_bp_rp'
   ------------------------------------------------------------------------
     HR -->      | 'grid_HR_to_mass'             | 'HR_to_mass'
     WD para.    | 'grid_HR_to_logg'             | 'HR_to_logg'
                 | 'grid_HR_to_age'              | 'HR_to_age'
                 | 'grid_HR_to_age_cool'         | 'HR_to_age_cool'
                 | 'grid_HR_to_logteff'          | 'HR_to_logteff'
                 | 'grid_HR_to_Mbol'             | 'HR_to_Mbol'
                 | 'grid_HR_to_cool_rate^-1'     | 'HR_to_cool_rate^-1'
   ------------------------------------------------------------------------
     others      |                               | 'm_agecool_to_bprp'
                 |                               | 'm_agecool_to_G'
   ======================================================================== 
    cooling-track data points:
'mass_array':   1d-array. The mass of WD in unit of solar mass. I only 
                read one value for a cooling track, not tracking the 
                mass change.
'logg':         1d-array. in cm/s^2
'age':          1d-array. The total age of the WD in yr. Some are read
                directly from the cooling tracks, but others are 
                calculated by assuming an initial--final mass relation
                (IFMR) of the WD and adding the rough main-sequence age
                to the cooling age.
'age_cool':     1d-array. The cooling age of the WD in yr.
'logteff':      1d-array. The logarithm effective temperature of the WD
                in Kelvin (K).
'Mbol':         1d-array. The absolute bolometric magnitude of the WD. 
                Many are converted from the log(L/Lsun) or log(L), where
                I adopt:
                        Mbol_sun = 4.75,
                        Lsun = 3.828e33 erg/s.
'cool_rate^-1': 1d-array. The reciprocal of cooling rate dt / d(bp-rp),
                in Gyr/mag.
        'G':            1d-array. The absolute magnitude of Gaia G band,
                        converted from the atmosphere interpolation.
        'bp_rp':        1d-array. The Gaia color index BP-RP, converted from the
                        atmosphere interpolation.






It reads the table of synthetic colors (http://www.astro.umontreal.ca/~bergeron/CoolingModels/),
generates the values of the Gaia color BP-RP and the bolometric correction
G-Mbol of Gaia G band, and interpolates the mapping:
        (logteff, logg) --> BP-RP or G-Mbol.
Then, it reads the mass, logg, total age (if it exists), cooling age,
logteff, and absolute bolometric magnitude Mbol from white dwarf cooling
models, and stacks the data points from different cooling tracks. 
  Finally, it interpolates the mapping between parameters such as logg, 
logteff and Gaia photometry. A typical form of these mappings is:
        (BP-RP, G) --> para,
from the HR diagram to WD parameters.




