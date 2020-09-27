import numpy as np
from perses.analysis.utils import open_netcdf
from tqdm import tqdm
import mdtraj as md

htf = np.load('outhybrid_factory.npy.npz',allow_pickle=True)
htf = htf['arr_0']
htf = htf.flatten()[0]

phase = 'complex'


for phase in ['complex','solvent']:
    # files that are needed
    # checkpoint file contains all the positions for the solvent molecules
    nc_checkpoint = open_netcdf(f'out-{phase}_checkpoint.nc')
    checkpoint_interval = nc_checkpoint.CheckpointInterval
    
    all_positions = nc_checkpoint.variables['positions'] # all of the positions of the hybrid system
    # need the normal .nc file too, as the checkpoint doesn't hold the state indices
    # this is needed to ascertain which are the endstates
    nc = open_netcdf(f'out-{phase}.nc')
    
    n_iter, n_replicas, _, _ = np.shape(all_positions)
    
    # getting the mapping from the hybrid-system to the new one
    hybrid_to_old = htf[f"{phase}"]._hybrid_to_old_map
    hybrid_to_new = htf[f"{phase}"]._hybrid_to_new_map
    ordered_old_ids = [hybrid for (hybrid, real) in sorted(hybrid_to_old.items(), key=lambda x: x[1])] 
    ordered_new_ids = [hybrid for (hybrid, real) in sorted(hybrid_to_new.items(), key=lambda x: x[1])] 
    
    # print(ordered_old_ids)
    
    n_old_atoms = len(hybrid_to_old)
    n_new_atoms = len(hybrid_to_new)
    
    # arrays to store the positions
    old_positions = np.zeros(shape=(n_iter,n_old_atoms,3))
    new_positions = np.zeros(shape=(n_iter,n_new_atoms,3))
    
    cell_lengths_old = np.zeros(shape=(n_iter,3))
    cell_lengths_new = np.zeros(shape=(n_iter,3))
    
    old_index = 0
    new_index = n_replicas - 1
    
    for iteration in tqdm(range(n_iter)):
        replica_id_old = np.where(nc.variables['states'][iteration*checkpoint_interval] == old_index)[0]
        pos = all_positions[iteration,replica_id_old,:,:][0]
        for hybrid, index in hybrid_to_old.items():
            old_positions[iteration,index,:] = pos[hybrid]
            box_lengths = [10*nc_checkpoint.variables['box_vectors'][iteration][replica_id_old[0]][j][j] for j in range(3)]
            cell_lengths_old[iteration,:] = box_lengths  
        replica_id_new = np.where(nc.variables['states'][iteration*checkpoint_interval] == new_index)[0]
        pos = all_positions[iteration,replica_id_new,:,:][0]
        for hybrid, index in hybrid_to_new.items():
            new_positions[iteration,index,:] = pos[hybrid]
            box_lengths = [10*nc_checkpoint.variables['box_vectors'][iteration][replica_id_new[0]][j][j] for j in range(3)]
            cell_lengths_new[iteration,:] = box_lengths
        
        
    TPs = np.load('out_topology_proposals.pkl',allow_pickle=True)
    TPs[f'{phase}_topology_proposal']._old_topology
    TPs[f'{phase}_topology_proposal']._new_topology
    
    
    md_top = md.Topology.from_openmm(TPs[f'{phase}_topology_proposal']._old_topology)
    traj = md.Trajectory(old_positions[0],md_top)
    traj.save_pdb(f'old_{phase}.pdb')
    
    with md.formats.DCDTrajectoryFile(f'old_positions_{phase}.dcd','w') as f:
        f.write(np.multiply(old_positions,10),cell_lengths=cell_lengths_new,cell_angles=np.ones(np.shape(cell_lengths_old))*90)
    
    md_top = md.Topology.from_openmm(TPs[f'{phase}_topology_proposal']._new_topology)
    traj = md.Trajectory(new_positions[0],md_top)
    traj.save_pdb(f'new_{phase}.pdb')
    
    with md.formats.DCDTrajectoryFile(f'new_positions_{phase}.dcd','w') as f:
        f.write(np.multiply(new_positions,10),cell_lengths=cell_lengths_new,cell_angles=np.ones(np.shape(cell_lengths_old))*90)
