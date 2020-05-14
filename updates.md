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
