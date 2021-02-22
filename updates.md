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
