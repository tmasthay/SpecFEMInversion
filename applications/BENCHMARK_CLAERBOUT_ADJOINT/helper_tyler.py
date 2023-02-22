import numpy as np
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
from subprocess import check_output as co
import sys
import re
from glob import *
import os

class ht:
    class MathTextSciFormatter(mticker.Formatter):
        def __init__(self, fmt="%1.2e"):
            self.fmt = fmt
        def __call__(self, x, pos=None):
            s = self.fmt % x
            decimal_point = '.'
            positive_sign = '+'
            tup = s.split('e')
            significand = tup[0].rstrip(decimal_point)
            sign = tup[1][0].replace(positive_sign, '')
            exponent = tup[1][1:].lstrip('0')
            if exponent:
                exponent = '10^{%s%s}' % (sign, exponent)
            if significand and exponent:
                s =  r'%s{\times}%s' % (significand, exponent)
            else:
                s =  r'%s%s' % (significand, exponent)
            return "${}$".format(s)

    def sn_plot(local_ax=plt, num_decimals=1):
        local_ax.gca().yaxis.set_major_formatter(ht.MathTextSciFormatter('%' + '1.%de'%num_decimals))

    def get_params(filename='DATA/Par_file_ref', type_map=dict()):
        try:
            d = dict()
            s = [re.sub('#.*$', '', e).strip() for e in open(filename, 'r').readlines()]
            s = [re.sub('[ ]+', ' ', e).strip() for e in s if e != '']
            for e in s:
                v = [ee.strip() for ee in e.split('=') if ee != '']
                if( len(v) == 1 ):
                    if( 'NO_FIELD' in d.keys() ): d['NO_FIELD'].append(v[0])
                    else: d['NO_FIELD'] = [v[0]]
                elif( len(v) == 2 ):
                    tm = str
                    if( v[0] in type_map.keys() ): tm = type_map[v[0]]
                    if( v[0] in d.keys() ): d[v[0]].append(tm(v[1]))
                    else: d[v[0]] = [tm(v[1])]
            return d
        except Exception as e:
            raise

    def par_pull(filename='DATA/Par_file_ref'):
        par_map = {'NSTEP': int,
            'DT': float,
            'NSOURCES': int,
            'NPROC': int,
            'SIMULATION_TYPE': int,
            'nrec': int,
            'xdeb': float,
            'xfin': float,
            'zdeb': float,
            'zfin': float
        }
        return ht.get_params(filename, par_map)

    def src_pull(filename='DATA/SOURCE'):
        tmp1 = lambda x: float(x.replace('d','e'))
        src_map = {'f0': float,
            'source_type': int,
            'time_function_type': int,
            'Mxx': tmp1,
            'Mzz': tmp1,
            'Mxz': tmp1,
            'factor': tmp1,
            'xs': float,
            'zs': float
        }
        return ht.get_params(filename, src_map)

    def create_ricker_time_derivative(base_dir='DATA', warn=False):
        if( base_dir[-1] == '/' ):
            base_dir = base_dir[:-1]

        par_fields = ht.par_pull(base_dir + '/Par_file_ref')
        source_fields = ht.src_pull(base_dir + '/SOURCE')

        nt = par_fields['NSTEP'][0]
        dt = par_fields['DT'][0]
        t = np.linspace(0.0, dt*(nt-1), nt)

        freq = source_fields['f0']
        for (i,f) in enumerate(freq):
            tmp1 = -6.0 * np.pi**2 * f * t**2
            tmp2 = 4.0 * np.pi**4 * f**3 * t**4
            tmp3 = np.exp(-np.pi**2 * f**2 * t**2)
            curr = (tmp1 + tmp2) * tmp3
            np.save('%s/ricker_time_deriv_%d.bin'%(base_dir, i), curr)

    def add_artificial_receivers(src, filename='DATA/Par_file', dz=1.0, dx=1.0):
        sp = lambda a,b: '%s%s= %s\n'%(a,28 * ' ',b)
        N = 3
        s = ''
        for i in range(-1,N-1):
            s += '# ARTIFICIAL RECEIVER GROUP %d\n'%(i+2)
            s += sp('nrec', str(N))
            s += sp('xdeb', '%.1f'%(src[0]-dx))
            s += sp('zdeb', '%.1f'%(src[1]+i*dz))
            s += sp('xfin', '%.1f'%(src[0]+dx))
            s += sp('zfin', '%.1f'%(src[1]+i*dz))
            s += 'record_at_surface_same_vertical = .false.\n\n'
        start_tag = '# ARTIFICIAL RECEIVERS START'
        f = open(filename, 'r')
        text = f.read()
        f.close()
        text = text.replace(start_tag, start_tag + '\n' + s)
        print(re.findall(r'%s'%start_tag, text))
        f = open(filename, 'w')
        f.write(text)
    
    def update_source(xs, zs, filename='DATA/SOURCE'):
        f = open(filename, 'r')
        s = f.read()
        f.close()
        t = 28 * ' '
        s = re.sub('xs.*=.*', 'xs%s= %.1f'%(t,xs), s)
        s = re.sub('zs.*=.*', 'zs%s= %.1f'%(t,zs), s)
        f = open(filename, 'w')
        f.write(s)
        f.close()

if( __name__ == "__main__" ):
    mode = int(sys.argv[1])
    if( mode == 1 ):
        ht.create_ricker_time_derivative('ELASTIC/DATA')
        par_fields = ht.par_pull('ELASTIC/DATA/Par_file_ref')
        nt = par_fields['NSTEP'][0]
        dt = par_fields['DT'][0]
        t = np.linspace(0.0, dt * (nt-1), nt)

        source_params = ht.src_pull('ELASTIC/DATA/SOURCE')
        freq = source_params['f0']

        v = [np.load(e) for e in glob('ELASTIC/DATA/ricker_time_deriv_[0-9]*.bin*')]

        for (i,e) in enumerate(v):
            plt.plot(t, e, label='%.1f'%freq[i])
            plt.savefig('%d.pdf'%i)

        plt.legend()
        plt.title('Frequency comparison')
        plt.savefig('freq.pdf')
    elif( mode == 2 ):
        v = ht.src_pull()
        src = [v['xs'][0], v['zs'][0]]
        ht.add_artificial_receivers(src)
    elif( mode == 3 ):
        xs = float(sys.argv[2])
        zs = float(sys.argv[3])
        ht.update_source(xs,zs)



