import numpy as np
import os
from netCDF4 import Dataset

def get_positions(ncfile, iteration, replica):
    # find the replica index of the lambda value
    replica_id = np.where(ncfile.variables['states'][iteration] == replica)[0]
    return ncfile.variables['positions'][iteration,replica_id,:,:][0]


def check_CNOH(string):
    for letter in string:
        if letter not in ['C', 'N', 'O', 'H']:
            return False
    return True

def store_positions(directory):
    directory = directory.strip('/')
    ligandA = directory[3:].split('to')[0]
    ligandB = directory[3:].split('to')[1]
    TPs = np.load(f'{directory}/out_topology_proposals.pkl',allow_pickle=True)
    htf = np.load(f'{directory}/outhybrid_factory.npy',allow_pickle=True)
    htf = htf.flatten()[0]
    
    phases = ['complex', 'solvent', 'vacuum']
    
    for phase in phases:
        filename = f'{directory}/out-{phase}.nc'
        if os.path.isfile(filename): # check it exists
            ncfile = Dataset(filename, 'r')
        else:
            print('ncfile doesn\t exist')
    
        n_iterations, n_replicas, _, _ = np.shape(ncfile.variables['positions'])

        endstates = [(ligandA,'old',0),(ligandB,'new',n_replicas-1)]
        for endstate in endstates:
            lig, state, index = endstate  
            topology = getattr(TPs[f'{phase}_topology_proposal'], f'{state}_topology')
            molecule = [res for res in topology.residues() if res.name == 'MOL']
            molecule_indices = [a.index for a in molecule[0].atoms()]
            n_atoms = len(molecule_indices)
            if index == 0:
                # need to get the start_id from the old system, not the new
                start_id = molecule_indices[0]
            string = ''.join([a.element.symbol for a in molecule[0].atoms()])
            if check_CNOH(string):
                positions = np.zeros(shape=(n_iterations,n_atoms,3))
                
                for i in range(n_iterations):
                    pos = get_positions(ncfile, i, index) 
                    for hybrid, old in getattr(htf['vacuum'], f'_hybrid_to_{state}_map').items():
                        positions[i,old,:] = pos[hybrid+start_id]
                
                np.save(f'{directory}/positions_{lig}_{phase}', positions)
            else:
                print(f'ligand {lig} contains atoms that aren\t ANI supported at the moment (CNOH).')
