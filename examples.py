import WD_models

# load the BaSTI models with and without phase separation
HR_grid = (-0.6, 1.25, 0.002, 10, 15, 0.01)
model_1  = WD_models.load_model('f', 'b', 'b', 'DA_thick', ('bp-rp', 'rp'), HR_grid) 
model_2  = WD_models.load_model('f', 'bn', 'bn', 'DA_thick', ('bp-rp', 'rp'), HR_grid)

plt.figure(figsize=(6,5),dpi=100)

# plot the mass contour
CS = plt.contour(model_1['grid_HR_to_mass'].T,
                 levels=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.19], 
                 extent=(HR_grid[0],HR_grid[1],HR_grid[3],HR_grid[4]),
                 aspect='auto', origin='lower', cmap='jet', vmin=-1000,
                 )
plt.clabel(CS, inline=True, use_clabeltext=True)

# plot the slowing-down effect of phase separation
contrast = ((model_1['grid_HR_to_cool_rate^-1'] - model_2['grid_HR_to_cool_rate^-1']) /
            (model_2['grid_HR_to_cool_rate^-1'])).T
plt.contourf(contrast, 
             levels=[0.00,0.10,0.20,0.30,0.40,0.50],
             extent=(HR_grid[0],HR_grid[1],HR_grid[3],HR_grid[4]),
             aspect='auto', origin='lower', cmap='binary', vmin=0.05, vmax=0.65
            )
plt.colorbar()

# plot the legend
plt.plot([0,0],[1,1],'-', color='gray', lw=10, 
         label='$\\Delta\\zeta/\\zeta_0$,\n' +
                'where $\\zeta=$d$t$/d(bp-rp) is\n' + 
                'the inverse of cooling rate')
plt.legend()

# set the figure
plt.title('The relative effect of phase separation (p.s.)\n' +
          'on WD cooling, calculated from the BaSTI\n' + 
          'DA models with and without p.s.')
plt.xlabel('BP - RP')
plt.ylabel('$\\rm M_G$')
plt.xlim(-0.6,1.25)
plt.ylim(15,10)
plt.show()
