import numpy as np

def sobolev(f, s=0, **kw):
    if( 'sample' in kw.keys() ):
        ot = kw['sample'][0]
        dt = kw['sample'][1]
        nt = kw['sample'][2]
    else:
        ot = kw['ot']
        dt = kw['dt']
        nt = kw['nt']
    xi = np.fft.fftfreq(nt, d=dt)
    f_hat = np.exp(-2j * np.pi * ot * xi) * np.fft.fft(f) * dt

    xi = np.fft.fftshift(xi)
    f_hat = np.fft.fftshift(f_hat)
    g = (1 + np.abs(xi)**2)**s * np.abs(f_hat)**2
    dxi = xi[1] - xi[0]
    res = np.trapz(g, dx=dxi)
    return res, xi, g

