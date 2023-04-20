import numpy as np
from scipy.integrate import cumulative_trapezoid
from typlotlib import *
import matplotlib.pyplot as plt
from smart_quantile_cython import *

x = np.linspace(-4 * np.pi, 4 * np.pi, 1000)
dx = x[1] - x[0]
x0 = 5.0

def h(y, alpha=1.0):
    u = (1.0 + np.cos(x)) * np.exp(-alpha * y**2)
    return u / np.trapz(u, dx=y[1]-y[0])

beta = 0.001
omega = 50.0
f = h(50.0*(x+x0), alpha=beta)
g = h(50.0*(x-x0), alpha=beta)

f = f / np.trapz(f, dx=dx)
g = g / np.trapz(g, dx=dx)

F = cumulative_trapezoid(f, dx=dx, initial=0.0)
G = cumulative_trapezoid(g, dx=dx, initial=0.0)

p = np.linspace(0,1,1000)

f = np.array(f, dtype=np.float32)
g = np.array(g, dtype=np.float32)
F = np.array(F, dtype=np.float32)
G = np.array(G, dtype=np.float32)
x = np.array(x, dtype=np.float32)
p = np.array(p, dtype=np.float32)


QF = smart_quantile(x, f, F, p)
QG = smart_quantile(x, g, G, p)

scp_dummy = set_color_plot_global(use_legend=True)
scp = lambda title : scp_dummy(title, 'white')

c1 = 'dodgerblue'
c2 = 'darkorange'
c3 = 'mediumspringgreen'
c4 = 'yellow'
c5 = 'mediumslateblue'
setup_gg_plot('black', 'black')
plt.plot(x, f, label=r'$f$', linestyle='-', color=c1)
plt.plot(x, g, label=r'$g$', linestyle='-', color=c2)
scp('Normalized data (PDF)')
plt.savefig('density.pdf')
plt.close()

plt.plot(
    x, 
    F, 
    linestyle='-', 
    color=c1, 
    label=r'$F(x) = \int_{-\infty}^{x} f(t) dt$'
)
plt.plot(
    x, 
    G, 
    linestyle='-', 
    color=c2, 
    label=r'$G(x) = \int_{-\infty}^{x} g(t)dt$'
)
scp('Integrated data (CDF)')
plt.savefig('cumulative.pdf')
plt.close()

add_diff = False
add_densities = False
add_cdfs = True
add_quantiles = False
add_quant_area = False
make_squared = False
add_quant_diff = False
add_cdf_area = True
if( add_diff ):
    plt.plot(
        x, 
        np.abs(F-G), 
        linestyle='-', 
        color=c3,
        label=r'$F-G$'
    )
    plt.fill_between(x, np.abs(F-G), alpha=0.3, color=c3)
if( add_densities ):
    plt.plot(
        x,
        f,
        linestyle=':',
        color=c1,
        label=r'$f$'
    )
    plt.plot(
        x,
        g,
        linestyle=':',
        color=c2,
        label=r'$g$'
    )
if( add_cdfs ):
    plt.plot(
        x,
        F,
        linestyle='-.',
        color=c4,
        label=r'$F$'
    )
    plt.plot(
        x,
        G,
        linestyle='-.',
        label=r'$G$',
        color=c5
    )
    if( add_cdf_area ):
        plt.fill_between(f,F,G,alpha=0.3,color=c3)
if( add_quantiles ):
    plt.plot(
        p,
        QF,
        linestyle='-.',
        color=c4,
        label=r'$F^{-1}$',
    )
    plt.plot(
        p,
        QG,
        linestyle='-.',
        color=c5,
        label=r'$G^{-1}$'
    )
    if( add_quant_area ):
        plt.fill_between(p, QF, QG, alpha=0.3, color=c3)
if( add_quant_diff ):
    u = np.abs(QF - QG)
    if( make_squared ):
        u = u**2
        plt.plot(
            p,
            u,
            color=c3,
            label=r'$|F^{-1}-G^{-1}|^2$'
        )
    else:
        plt.plot(
            p,
            u,
            color=c3,
            label=r'$|F^{-1}-G^{-1}|$'
        )
    if( add_quant_area ):
        plt.fill_between(p, u, alpha=0.3, color=c3)

scp(r'Relationship between $W_2$ and $\|\cdot\|_{H^{-1}}$')
plt.savefig('cumulative_diff.pdf')