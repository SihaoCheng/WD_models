# WD_models
I provide a python module for transformation between the *Gaia* H--R diagram and many white dwarf (WD) physical parameters (L, Teff, log *g*, etc), based on an interpolation of cooling tracks from various existing WD models (listed below). This module is written for python 3 and will use Functions from the following packages: `astropy, matplotlib, numpy, scipy`. It is designed mainly for the following purposes:

1. converting the coordinates of *Gaia* H--R diagram into WD parameters;
2. plotting the contour of WD parameters on the *Gaia* H--R diagram.

However, one can also achieve conversions between any desired WD parameters easily (see Example 2 below), based on the tools provided in this module.


## Import
Please download the script `WD_models.py` and folder `models/` to the same directory, and simply import the module in python 3:
```python
import WD_models
```

## Example 1: converting H--R diagram coordinate into WD parameters
```python
model = WD_models.load_model(low_mass_model='Fontaine2001',
                             normal_mass_model='Althaus2010_001',
                             high_mass_model='ONe',
                             spec_type='DA_thick',
                             )
                             
# the cooling age at (BP-RP, G) = (0.25, 13) and (0.25, 14)
age_cool = model['HR_to_age_cool']([0.25, 0.25], [13,14])

print(age_cool)
>> array([ 1.27785237,  2.70284467])
```
The outputs are in unit of Gyr. The *Function* `load_model` in the module reads a set of cooling tracks assigned by the user and returns a dictionary containing many useful functions for parameter conversion. The keys of this dictionary (available functions for other parameters) are listed in table ? below.

## Example 2: conversions between any desired WD parameters

If the conversion of a desired conversion is not provided in the output of `load_model`, the user can generate the interpolated grid values and mapping function with the function `interp_xy_z_func`, `interp_xy_z_func`, or `interp_HR_to_para`, based on the cooling-track data points and atmosphere grid provided as the output of `load_model`. 

For example, for the mapping (mass, logteff) --> cooling age:
```python
model = WD_models.load_model('f', 'a001', 'o', 'DA_thick')

m_logteff_to_agecool = WD_models.interp_xy_z_func(
    model['mass_array'], model['logteff'], model['age_cool'], 'linear')
    
age_cool = m_logteff_to_agecool(1.1, np.log10(10000))

print(age_cool)
>> 2.1926053524257165
```
Note that there are shorter versions for the names of WD models, which are also listed in table ? below.

## Example 3: comparing different models
```python
model_A = WD_models.load_model('', 'b', 'b', 'DA_thick')
model_B = WD_models.load_model('', 'b', 'b', 'DB')

d_age_cool = (model_A['HR_to_age_cool'](0, 13) - 
              model_B['HR_to_age_cool'](0, 13))

print(d_age_cool)
>> 0.274022022781
```

## Available models included in this module

### Low-mass models (less than about 0.5 Msun)

model names | remarks & reference
------------|----------------------
''                              |no low-mass model will be read
'Fontaine2001' or 'f'           |http://www.astro.umontreal.ca/~bergeron/CoolingModels/

### Normal-mass models (about 0.5 to 1.0 Msun)

model names | remarks & reference
------------|----------------------
''                              |no normal-mass model will be read
'Fontaine2001' or 'f'           |http://www.astro.umontreal.ca/~bergeron/CoolingModels/
'Althaus2010_001' or 'a001'     |Z=0.01, only for DA, http://evolgroup.fcaglp.unlp.edu.ar/TRACKS/tracks_cocore.html
'Althaus2010_0001' or 'a0001'   |Z=0.001, only for DA, http://evolgroup.fcaglp.unlp.edu.ar/TRACKS/tracks_cocore.html
'Camisassa2017' or 'c'          |only for DB, http://evolgroup.fcaglp.unlp.edu.ar/TRACKS/tracks_DODB.html
'BaSTI' or 'b'                  |with phase separation, Salaris et al. 2010, http://basti.oa-teramo.inaf.it
'BaSTI_nosep' or 'bn'           |no phase separation, Salaris et al. 2010, http://basti.oa-teramo.inaf.it
'PG'                            |only for DB

### High-mass models (higher than 1.0 Msun)

model names | remarks & reference
------------|----------------------
''                              |no high-mass model will be read
'Fontaine2001' or 'f'           |http://www.astro.umontreal.ca/~bergeron/CoolingModels/
'ONe' or 'o'                    |Camisassa et al. 2019, http://evolgroup.fcaglp.unlp.edu.ar/TRACKS/ultramassive.html
'MESA' or 'm'                   |Lauffer et al. 2019
'BaSTI' or 'b'                  |with phase separation, Salaris et al. 2010, http://basti.oa-teramo.inaf.it
'BaSTI_nosep' or 'bn'           |no phase separation, Salaris et al. 2010, http://basti.oa-teramo.inaf.it

### Spectral type and atmosphere sythetic colors

We interpolate the table of sythetic colors from http://www.astro.umontreal.ca/~bergeron/CoolingModels/). The spectral type can be one of the following:

spectral type | remarks
--------------|----------------------
'DA_thick'    | thick hydrogen atmosphere
'DA_thin'     | thin hydrogen atmosphere
'DB'          | pure-helium atmosphere
     
## Output of the function `load_model`

The function `load_model` returns a dictionary, which contains the atmosphere grids and mapping, parameter mappings, and stacked cooling-track data points. The keys of this dictionary are:

### Interpolation results

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

### Cooling-track data points

name | remarks
-----|---------
'mass_array':   | 1d-array. The mass of WD in unit of solar mass. I only read one value for a cooling track, not tracking the mass change.
'logg':         | 1d-array. in cm/s^2
'age':          | 1d-array. The total age of the WD in yr. Some are read directly from the cooling tracks, but others are calculated by assuming an initial--final mass relation (IFMR) of the WD and adding the rough main-sequence age to the cooling age.
'age_cool':     | 1d-array. The cooling age of the WD in yr.
'logteff':      | 1d-array. The logarithm effective temperature of the WD in Kelvin (K).
'Mbol':         | 1d-array. The absolute bolometric magnitude of the WD. Many are converted from the log(L/Lsun) or log(L), where I adopt: Mbol_sun = 4.75, Lsun = 3.828e33 erg/s.
'cool_rate^-1': | 1d-array. The reciprocal of cooling rate dt / d(bp-rp), in Gyr/mag.
'G':            | 1d-array. The absolute magnitude of Gaia G band, converted from the atmosphere interpolation.
'bp_rp':        | 1d-array. The Gaia color index BP-RP, converted from the atmosphere interpolation.



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




