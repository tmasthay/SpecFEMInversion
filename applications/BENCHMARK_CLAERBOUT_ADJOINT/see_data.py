import numpy as np
from helper_tyler import *
from helper_functions import helper
import matplotlib.pyplot as plt
import time 

u = ht.sco('find ELASTIC -type d -name "convex_[0-9]*_[0-9]*"', True)

x_max = max(list(set([int(e.split('_')[1]) for e in u])))
z_max = max(list(set([int(e.split('_')[2]) for e in u])))

ax = 1000
bx = 3000
az = 500
bz = 2500

stride_x = 2.5
stride_z = 2.5

x_max = int(x_max / stride_x)
z_max = int(z_max / stride_z)
x = np.linspace(ax, bx, x_max)
z = np.linspace(az, bz, z_max)

hf = helper()

plt.rcParams['text.usetex'] = True
k = 0

t_min = 0.0
t_max = 1600 * 1.1e-3

total_tasks = x_max * z_max
img = plt.imread('Velocity.jpg')
t = time.time()
for iz in range(z_max):
    for ix in range(x_max-1,-1,-1):
        ux = hf.read_SU_file('ELASTIC/convex_%d_%d/Ux_file_single_d.su'%(ix,iz))
        uz = hf.read_SU_file('ELASTIC/convex_%d_%d/Uz_file_single_d.su'%(ix,iz))

        fig = plt.figure(constrained_layout=True, figsize=(20,10))
        axs = fig.subplot_mosaic(
            [['Source', 'Source'], ['ux', 'uz']],
            gridspec_kw={'width_ratios': [1,1], 'height_ratios': [1,1]},
        )
        axs['Source'].imshow(img, extent=[0,4000,0,3000], aspect='auto')
        axs['Source'].scatter(
            [x[ix]],
            [z[iz]],
            color='red',
            s=100,
            marker='*'
        )

        axs['ux'].imshow(
            ux, 
            extent=[ax,bx,t_min,t_max], 
            aspect='auto',
            cmap='jet'
        )
        axs['ux'].set_title(r'$u_x$ for $(s_x,s_z)=(%.1f,%.1f)$'%(x[ix], z[iz]))
        axs['ux'].set_xlabel(r'Offset $x$ (km)')
        axs['ux'].set_ylabel(r'Time (s)')

        axs['uz'].imshow(
            uz, 
            extent=[ax,bx,t_min,t_max], 
            aspect='auto',
            cmap='jet'
        )
        axs['uz'].set_title(r'$u_z$ for $(s_x,s_z)=(%.1f,%.1f)$'%(x[ix], z[iz]))
        axs['uz'].set_xlabel(r'Offset $x$ (km)')
        axs['uz'].set_ylabel(r'Time (s)')

        plt.savefig('data_%d.jpg'%(k))
        plt.close()
        k += 1
        total_time = time.time() - t
        avg_time = total_time / k
        print('%d/%d in %f seconds...%f seconds remaining'%(
                k,
                total_tasks, 
                total_time, 
                avg_time * (total_tasks - k)
            )
        )

os.system('convert -delay 100 -loop 0 $(ls -t data_*.jpg) output.gif')
os.system('open output.gif')

