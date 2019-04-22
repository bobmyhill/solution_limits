import numpy as np
from sympy import Matrix, ones, zeros
import copy
import argparse

parser = argparse.ArgumentParser(description='Calculate the limits on the ordered endmembers within a solution.')
parser.add_argument("fname", 
                    help='The path to the input file. For formatting of this file, see the examples in the examples directory')

args = parser.parse_args()
fname = args.fname


with open(fname) as f:
    data = f.readlines()
    data = [i for i in [line.strip().split() for line in data] if i and i[0] != '%']

sites = data[0]
multiplicities=Matrix(data[1])
n_disordered = data[2].index('|')
n_ordered = len(data[2]) - n_disordered - 1

names_disordered = data[2][0:n_disordered]
names_ordered = data[2][n_disordered+1:]

names = copy.deepcopy(names_disordered)
names.extend(names_ordered)

atoms = [d[0] for d in data[3:]]
atomic_sites = [d[1] for d in data[3:]]
unique_atoms = list(set(atoms))
atom_indices = [[i for i, a in enumerate(atoms) if a==atom] for atom in unique_atoms]

atomic_site_multiplicities = Matrix([Matrix([multiplicities[sites.index(site)]
                                            for site in atomic_sites]).T for i in range(len(data[3]) - 2)])
site_fractions = Matrix([d[2:] for d in data[3:]])
atom_site_matrix = site_fractions.T.multiply_elementwise(atomic_site_multiplicities)

atom_matrix = Matrix([ones(1, len(atom_idxs))*atom_site_matrix[:,atom_idxs].T
                      for atom_idxs in atom_indices])

n_atoms = len(unique_atoms)
n_atoms_per_site = np.array([len([i for i, s in enumerate(atomic_sites) if s==site]) for site in sites])

A = atom_matrix[:,0:n_disordered]
if n_disordered < n_atoms:
    A = A.row_join(ones(n_atoms, n_atoms - n_disordered))

b = atom_matrix[:,n_disordered:]

c = A.LUsolve(b)[0:n_disordered,:]

print('{0}                            | {0} ordered species:'.format(n_ordered))
print('')
for j in range(n_ordered):
    string='{0} = '.format(names_ordered[j])
    first=True
    for i, name in enumerate(names_disordered):
        if c[i,j] > 0:
            if first:
                string='{0} {1} {2}'.format(string, c[i,j], name)
                first=False
            else:
                string='{0} + {1} {2}'.format(string, c[i,j], name)
        elif c[i,j] < 0:
            string='{0} - {1} {2}'.format(string, -c[i,j], name)
            first=False
    print(string)
print('')

c = c.col_join(zeros(n_ordered, n_ordered))
d = site_fractions*c

def append_to_string(string, f, name):
    if f > 0:
        string = '{0} + {1} {2}'.format(string, f, name)
    elif f < 0:
        string = '{0} - {1} {2}'.format(string, -f, name) 
    return string


n_site_limits = []
print('begin_limits')
#loop over ordered phases
for j in range(n_ordered):
    n_site_limits.append([0]*len(sites))
    if j != 0:
        print('')
    # loop over atomic sites
    for i in range(len(atomic_sites)):
        pord = site_fractions[i,n_disordered+j] - d[i,j]
        if pord != 0:
            n_site_limits[-1][sites.index(atomic_sites[i])] += 1
            
            delta = -1/pord
            s = site_fractions[i,n_disordered+j]

            string=''
            for k in range(n_disordered):
                string = append_to_string(string, delta*site_fractions[i,k], names[k])
                
            for k in range(n_ordered):
                if k == j:
                    string = append_to_string(string, delta*d[i,k], names[n_disordered+k]) 
                else:
                    string = append_to_string(string,
                                              delta*(site_fractions[i,n_disordered+k] - d[i,k]),
                                              names[n_disordered+k])
                    string = append_to_string(string,
                                              delta*d[i,k],
                                              names[n_disordered+k])
                    
            if delta > 0:
                string = '{0} = {1}{2}'.format(names_ordered[j], -delta, string)
            else:
                string = '{0} ={1}'.format(names_ordered[j], string)
                delta = -delta
                
            print('{0:65s} delta = {1} (limits for {2} on {3})'.format(string, delta, atoms[i], atomic_sites[i]))
            site_fractions[i,n_disordered+j] = s

print('end_limits')



print('')
print('Can remove 1 limit from each of the following listed ordered endmember sites:')
degrees_freedom = [[sites[i] for i, degrees_freedom in enumerate(n_atoms_per_site - np.array(ns)) if degrees_freedom == 0] for ns in n_site_limits]
for j, name in enumerate(names_ordered):
    print('{1}: {2}'.format(len(degrees_freedom[j]), name, degrees_freedom[j]))
    
