"""
The class PySCF is a wrapper of pyscf

(1) Potential energy of the QM molecule.
  1.A  An AtomGroup of the QM molecule is selected by
        qm_ag = u.select_atoms ('resname MOL')
  1.B  method is chosen from 'scf', 'mp2', 'svwn', 'bp86', ... 'b3lyp'
  1.C  basis should be provided by Users.
  1.D  charge is the total charge of the QM molecule.
------(example code)----
    u = mda.Universe(PDB, DCD)
    qm_ag = u.select_atoms('resname MOL')

    mda_pyscf = PySCF(qm_ag, method='scf',basis='sto-3g',charge=0)
    mda_pyscf.run(start=0, stop=4)
    etot = mda_pyscf.energy_tot()
    print('etot of QM', etot)
------(end)---

(2) Potential energy of the QM/MM molecule
 2.A the atomic charges of the MM system are needed.
   Here, pdb2charge uses the OpenMM (using force field amber14SB and tip3p) to
   estimate the atomic charges of the MM system (Whole system - QM system)

------(example code)----
    import pdb2charge
    chg_list = pdb2charge.pdb2charge (PDB_NO_QM)

    u = mda.Universe(PDB, DCD)
    qm_ag = u.select_atoms('resname MOL')

    mda_pyscf = PySCF(qm_ag, method='scf',basis='sto-3g',charge=0,
                      l_mm=True, mm_chg=chg_list, l_epol=True)
    mda_pyscf.run(start=0, stop=4)
    etot = mda_pyscf.energy_tot()
    epol = mda_pyscf.energy_pol()
    print('etot of QM/MM', etot)
    print('epol of QM embedded in MM', epol)
------(end)---
"""
from __future__ import absolute_import

import numpy as np
import warnings
import MDAnalysis as mda
from MDAnalysis.analysis.base import AnalysisBase
from pyscf.data.nist import BOHR
import sys


ang2bohr = 1.0/BOHR
bohr2ang = BOHR

dft_xc = ['svwn', 'bp86', 'blyp', 'pbe', 'lda,vwn_rpa',
          'b97,pw91', 'pbe0', 'b3p86', 'wb97x', 'b3lyp']


def _init_neigh_list(n_neigh):
    '''
    n_neigh: int
    generate neighboring cells with rcut_cell = rcut/n_neigh,
    rcut is the cutoff distance

    Returns
    -------
    np.array ([-n_neigh:n_neigh, -n_neigh:n_neigh, -n_neigh:n_neigh])
    '''
    neigh_list = []

    for ic in range(-n_neigh, n_neigh+1):
        for jc in range(-n_neigh, n_neigh+1):
            for kc in range(-n_neigh, n_neigh+1):
                neigh_list.append([ic, jc, kc])

    return np.array(neigh_list)


def _init_cell_list(ndim):
    '''
    ndim: np.array (3)
     the number of cell in x, y, z directions

    using a cell (linked) list method,

    Returns
    -------
    cell_list: dictionary
      each cell will store the atom lists
    '''

    cell_list = {}
    for ic in range(ndim[0]):
        for jc in range(ndim[1]):
            for kc in range(ndim[2]):
                cell_list[(ic, jc, kc)] = []

    return cell_list


def _pyscf_mol_build(qm_atm_list, basis_name, charge):
    """
    Generate the atom list in the QM region (pyscf.gto.Mole)

    Parameters
    ----------
    qm_atm_list : list of QM atoms with the following list
        [ ['sym1', (x1, y1, z1)], ['sym2', (x2,y2,z2)], ...]
        Note, the unit of position is in Bohr
    basis_name : string
    charge : int
        The total charge of the QM region

    Returns
    ------
    mol : pyscf.gto.Mole
    """
    from pyscf import gto

    mol = gto.Mole()
    mol.basis = basis_name
    mol.atom = qm_atm_list
    mol.charge = charge
    mol.unit = 'Bohr'
    mol.verbose = 0  # Turn off the print out
    mol.build()

    return mol


def _get_epol(mf_qmmm, mf_qm):
    """
    Calculate the polarization energy using
    Epol = <psi_qmmm | H_qmmm | psi_qmmm> - < psi_qm | H_qmmm | psi_qm>

    Parameters
    ----------
    mf_qmmm: an instance of pyscf.qmmm.QMMM class
        Here, mf_qmmm.e_tot will be used.
        mf_qmmm.e_tot = <psi_qmmm | H_qmmm | psi_qmmm>
    mf_qm: an instance of pyscf.scf.SCF class
        Here, mf_qm.make_rdm1 () will be used.
        mf_qm.make_rdm1() = <psi_qm| |psi_qm>

    Returns
    -------
    epol : float
        the polarization energy
    """
    # density matrix of the QM
    dm_qm = mf_qm.make_rdm1()
    # The second term indicates
    # the total energy with the electronic density:
    # dm_qm = <psi_qm| |psi_qm>
    epol = mf_qmmm.e_tot - mf_qmmm.energy_tot(dm_qm)

    return epol


class PySCF (AnalysisBase):

    def __init__(self, ag,
                 method='scf',
                 basis='sto-3g',
                 charge=0,
                 l_mm=False,
                 rcut=15,  # in A
                 mm_chg=None,
                 l_epol=False,
                 **kwargs):
        """
        Parameters
        ----------
        ag: AtomGroup or Universe
        method: theory or dft xc name, e.g. 'scf', 'mp2', ...'b3lyp'
        basis:  basis name, e.g. 'aug-cc-pVDZ', '6-311gss', ...
        charge: the total charge of the QM molecule
        l_mm: (bool) turn on/off the QM/MM calculation
        rcut: (float) the cutoff distance between QM and MM regions
        mm_chg: list or np.array (list) : MM atomic charges
        l_epol: (bool) turn on/off the polarization energy
        """

        super(PySCF, self).__init__(ag.universe.trajectory, **kwargs)

        self._ag = ag
        self.qm_method = method
        self.qm_basis = basis
        self.qm_charge = charge
        self.l_mm = l_mm
        self.mm_rcut = rcut
        self.mm_chg = mm_chg

        from MDAnalysis.topology.core import guess_atom_element
        self.symbols = []
        for atomname in ag.names:
            self.symbols.append(guess_atom_element(atomname.strip()))

        self.rcut = rcut*ang2bohr
        self.rcut_half = self.rcut/2
        self.l_epol = False

        if l_mm:
            if self.mm_chg is None:
                sys.exit()

            qm_idx = ag.indices
            selections = np.array(
                [ia for ia in range(ag.universe.atoms.n_atoms)
                 if ia not in qm_idx])

            self.mm_ag = mda.AtomGroup(selections, ag.universe)
            if self.mm_chg.shape[0] != self.mm_ag.indices.shape[0]:
                sys.exit()

            self.neigh_list = _init_neigh_list(2)

            if l_epol:
                self.l_epol = l_epol

    def _prepare(self):
        self.epol = []
        self.etot = []

    def _single_frame(self):
        from pyscf import scf, dft, mp, qmmm

        # (1) Coordinates of the QM molecule, in Bohr
        crd = self._ag.positions*ang2bohr

        # (2) Save [ symbol, (x, y, z)] into atm_list
        atm_list = []
        for ia, xyz in enumerate(crd):
            atm_list.append([self.symbols[ia], (xyz[0], xyz[1], xyz[2])])

        # (3) Build pyscf.Mole
        mol = _pyscf_mol_build(atm_list, self.qm_basis, self.qm_charge)

        # (4) Generate mf (mean field) according to method (or theory)
        if self.qm_method in ['scf', 'mp2']:
            mf = scf.HF(mol)
            if self.qm_method == 'mp2':
                mf = mp.MP2(mf)

        elif self.qm_method in dft_xc:
            mf = dft.KS(mol)
            mf.xc = self.method

        mf.run()
        e_tot = mf.e_tot
        e_pol = 0.0

        # (5) Check whether the system is QMMM
        if self.l_mm:
            rcut2 = self.rcut*self.rcut
            rcut_half = self.rcut_half
            mm_crd = self.mm_ag.positions*ang2bohr

            # (5.1) Check the pbc
            l_pbc = True
            if (self._ag.dimensions[:3] == 0).any():
                unitcell = mm_crd.max(axis=0) + 1.0
                l_pbc = False
            else:
                unitcell = self._ag.dimensions[:3]*ang2bohr

            # (5.2) Cell Linked List
            #       MM positions are stored into cell_list
            ndim = np.array(unitcell//self.rcut_half, dtype=np.int32)
            rcut_cell = unitcell/ndim

            cell_list = _init_cell_list(ndim)
            for ja, rj in enumerate(mm_crd):
                jcel = np.array(rj//rcut_cell, dtype=np.int32)
                ic, jc, kc = jcel
                rj0 = rj
                if (ic, jc, kc) not in cell_list:
                    #print('mm atoms outside unitcell', jcel)
                    ic, jc, kc = jcel - (jcel//ndim)*ndim
                    rj0 = rj - (jcel//ndim)*unitcell

                cell_list[(ic, jc, kc)].append([self.mm_chg[ja], rj0])

            # (5.3) Generate a cell_list that is within cutoff distance
            #       from the QM molecule
            qm_ncell_list = []
            for ia, ri in enumerate(crd):
                #icel = np.array(ri//self.rcut_half, dtype=np.int32)
                icel = np.array(ri//rcut_cell, dtype=np.int32)
                ncel = icel + self.neigh_list

                for ic, jc, kc in ncel:
                    if (ic, jc, kc) not in qm_ncell_list:
                        qm_ncell_list.append((ic, jc, kc))

            mm_cut = []
            qq_cut = []
            if l_pbc:
                for jcel in qm_ncell_list:
                    jmove = -(jcel//ndim)*ndim
                    jbox = (jcel//ndim)*unitcell
                    ic, jc, kc = jcel + jmove

                    for qj, rj0 in cell_list[(ic, jc, kc)]:  # mm_crd
                        rj = rj0+jbox
                        rij2_min = rcut2 + 100.0

                        for ri in crd:  # QM
                            dij = ri - rj
                            rij2 = np.einsum('i,i', dij, dij)
                            rij2_min = min(rij2, rij2_min)

                        if rij2_min < rcut2:
                            mm_cut.append(rj)
                            qq_cut.append(qj)

            else:
                for ic, jc, kc in qm_ncell_list:
                    if (ic, jc, kc) in cell_list:

                        for qj, rj in cell_list[(ic, jc, kc)]:
                            rij2_min = rcut2 + 100.0

                            for ri in crd:
                                dij = ri - rj
                                rij2 = np.einsum('i,i', dij, dij)
                                rij2_min = min(rij2, rij2_min)

                            if rij2_min < rcut2:
                                mm_cut.append(rj)
                                qq_cut.append(qj)

            mm_cut = np.array(mm_cut)
            qq_cut = np.array(qq_cut)
            mf_qmmm = qmmm.mm_charge(mf, mm_cut, qq_cut)
            mf_qmmm.run()
            e_tot = mf_qmmm.e_tot
            if self.l_epol:
                e_pol = _get_epol(mf_qmmm, mf)

            with open(f'QMMM{len(self.etot)}.xyz','w') as f:
                natom = len(atm_list) + len(mm_cut)
                for sym, xyz in atm_list:
                    f.write(f'{sym} {xyz[0]/ang2bohr} {xyz[1]/ang2bohr} {xyz[2]/ang2bohr}\n')
                for rj in mm_cut:
                    f.write(f'H {rj[0]/ang2bohr} {rj[1]/ang2bohr} {rj[2]/ang2bohr}\n')

        print('etot', self._frame_index, e_tot)
        print('epol', self._frame_index, e_pol)
        self.etot.append(e_tot)
        self.epol.append(e_pol)

    def energy_tot(self):

        return np.array(self.etot)

    def energy_pol(self):

        return np.array(self.epol)


if __name__ == '__main__':

    import pdb2charge
    import sys

    pdb = sys.argv[1]
    dcd = sys.argv[2]
    output = pdb.split('.')[0]

    results = {}
    chg_list = pdb2charge.pdb2charge(pdb)

    #u = mda.Universe('old_solvent.pdb', 'old_positions_solvent.dcd')
    u = mda.Universe(pdb, dcd)

    qm_ag = u.select_atoms('resname MOL')
    # (2) Test QM with MM and the polarization energy
    mda_pyscf = PySCF(qm_ag, method='scf', l_mm=True,
                      mm_chg=chg_list, l_epol=True)
    mda_pyscf.run(start=0, stop=100,verbose=True)
    etot = mda_pyscf.energy_tot()
    epol = mda_pyscf.energy_pol()
    results['QMMM'] = etot
    results['Epol'] = epol

    # (1) Test QM without MM
    mda_pyscf = PySCF(qm_ag, method='scf')
    mda_pyscf.run(start=0, stop=100,verbose=True)
    etot = mda_pyscf.energy_tot()
    results['QM'] = etot



    np.save(output,results) 
