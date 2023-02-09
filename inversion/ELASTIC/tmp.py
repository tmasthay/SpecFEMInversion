from helper_functions import helper
import matplotlib.pyplot as plt

u = helper(True)

base = 'MODELS/target_model/proc000000'
filename = base + '_NSPEC_ibool.bin'
f = open(filename, 'r')

nspec,ibool = u.read_binary_NSPEC_ibool_file(filename=filename)
vp = u.read_binary_SEM_file(filename=(base + '_rho.bin'))
vs = u.read_binary_SEM_file(filename=(base + '_vs.bin'))
x = u.read_binary_SEM_file(filename=(base + '_x.bin'))
z = u.read_binary_SEM_file(filename=(base + '_z.bin'))

plt.scatter(x,z)
plt.savefig('tmp.pdf')

