from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import elementary_charge
from math import floor
import numpy

def pdb2charge(fname_pdb):
    '''
    read pdb file and forcefield files and then
    return charges
    ---
    fname_pdb : string
    fname_ff  : [string, string, ..]
    return 
    chg_list : numpy.array ([q1, q2, ...])
    '''
    pdb = PDBFile (fname_pdb)
    ff  = ForceField ('amber14/protein.ff14SB.xml', 'amber14/tip3p.xml')
    
    m = Modeller(pdb.topology, pdb.positions)
    ligand = [r for r in pdb.topology.residues() if r.name == 'MOL']
    
    m.delete(ligand)
    
    sys = ff.createSystem(m.topology)


    for i in range (sys.getNumForces()):
        if isinstance(sys.getForce(i), NonbondedForce):
            nonbonded = sys.getForce(i)
            break

    
    chg_list = []

    for i in range(nonbonded.getNumParticles()):
        nb_i = nonbonded.getParticleParameters(i)
        chg  = nb_i[0].value_in_unit(elementary_charge)
        chg_list.append (chg)
    
    return numpy.array (chg_list)

if __name__ == "__main__":
    fname_pdb = 'old_solvent_nomol.pdb'
  
    chg_list = pdb2charge (fname_pdb)
    
    print (chg_list)
    
