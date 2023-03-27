from wass_v3 import *
import matplotlib.pyplot as plt
import argparse

t = 0.00140 * np.array(range(1500))
path = 'ELASTIC/convex_reference'
tau = 0.0
version = 'l2'
make_plots = True
regenerate = True
if( regenerate ):
    evaluators = create_evaluators(
        t, 
        path, 
        tau=tau, 
        version=version, 
        make_plots=make_plots
    )
    threaded = True
    if( threaded ):
        u = wass_landscape_threaded(evaluators)
    else:
        u = wass_landscape(evaluators)
    np.save('%s_landscape.npy'%version, u)

plt.rcParams['text.usetex'] = True 
plt.imshow(u, extent=[100,900,100,900], cmap='jet')
if( version == 'l2' ):
    plt.title(r'$L^2$')
elif( version == 'split' ):
    plt.title(r'Split Renormalized $W_2^2$')
else:
    plt.title(r'Square Renormalized $W_2^2$')
plt.xlabel('Horizontal Distance (km)')
plt.ylabel('Depth (km)')
plt.colorbar()
plt.savefig('%s_landscape.pdf'%version)
