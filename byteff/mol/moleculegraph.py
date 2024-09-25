# -----  ByteFF: ByteDance Force Field -----
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy as cp
from enum import Enum
from operator import attrgetter, itemgetter

import numpy as np
from rdkit import Chem

from byteff.mol import rkutil
from byteff.mol.molecule import Molecule
from byteff.mol.rkutil.match_and_map import find_mapped_smarts_matches
from byteff.mol.topology import Topology


class Hybridization(Enum):
    S = 0
    SP = 1
    SP2 = 2
    SP3 = 3
    SP3D = 4
    SP3D2 = 5
    OTHER = 6


hyb_map = {
    Chem.rdchem.HybridizationType.S: Hybridization.S,
    Chem.rdchem.HybridizationType.SP: Hybridization.SP,
    Chem.rdchem.HybridizationType.SP2: Hybridization.SP2,
    Chem.rdchem.HybridizationType.SP3: Hybridization.SP3,
    Chem.rdchem.HybridizationType.SP3D: Hybridization.SP3D,
    Chem.rdchem.HybridizationType.SP3D2: Hybridization.SP3D2,
}


class Atom:
    '''
    Tracks information about an atom
    '''

    def __init__(self, atom: Chem.Atom = None, idx: int = None):
        '''
        Initializes Atom based on a provided atom

        Parameters
        ----------
        atom : rdkit.Chem.Atom object or str
        idx : int
            The index of the atom in the molecule, only use for str atom input
        '''
        self.atom = atom
        self.coord = None
        self.atomic_number = None
        self.aromatic = None
        self.formal_charge = 0
        self.hydrogen_count = None

        # graph info
        self.connectivity = None
        self.ring_connectivity = None
        self.min_ring_size = None

        if atom is None:
            assert isinstance(idx, int) and idx >= 0
            self.idx = idx
        else:
            assert isinstance(atom, Chem.Atom)
            self.idx = atom.GetIdx()
            self.atomic_number = atom.GetAtomicNum()
            self.aromatic = atom.GetIsAromatic()
            self.formal_charge = round(atom.GetFormalCharge())
            self.hydrogen_count = atom.GetTotalNumHs(includeNeighbors=True)
            self.hybrdization = hyb_map.get(atom.GetHybridization(), Hybridization.OTHER)

    def __hash__(self) -> int:
        return self.idx

    def __lt__(self, other):
        return self.idx < other.idx

    def __eq__(self, other):
        if hasattr(other, 'idx'):
            return self.idx == other.idx
        return False

    def __str__(self):
        if self.atomic_number and self.connectivity:
            return f"[#{self.atomic_number}X{self.connectivity}]"
        else:
            return "[]"

    def get_copy(self):
        new_atom = Atom(idx=self.idx)
        for k, v in self.__dict__.items():
            if k[:2] == "__" or k == "atom":
                continue
            setattr(new_atom, k, cp.deepcopy(v))
        return new_atom


class Bond:
    '''
    Tracks information about a bond
    '''

    def __init__(self, bond: Chem.Bond = None, idx: int = None):
        '''
        Parameters
        ----------
        bond : rdkit.Chem.Bond or str
        idx: int
            the index of the bond in the molecule, only for str bond input
        '''
        self.bond = bond
        self.order = None
        self.is_conj = None
        self.begin_idx, self.end_idx = None, None

        # graph info
        self.in_ring = False

        if bond is None:
            assert isinstance(idx, int) and idx >= 0
            self.idx = idx
        else:
            assert isinstance(bond, Chem.Bond)
            self.idx = bond.GetIdx()
            self.order = bond.GetBondTypeAsDouble()
            self.is_conj = bond.GetIsConjugated()
            self.begin_idx = bond.GetBeginAtom().GetIdx()
            self.end_idx = bond.GetEndAtom().GetIdx()

    def __str__(self):
        return "~"

    def __hash__(self):
        return self.idx

    def get_copy(self):
        new_bond = Bond(idx=self.idx)
        for k, v in self.__dict__.items():
            if k[:2] == "__" or k == "bond":
                continue
            setattr(new_bond, k, cp.deepcopy(v))
        return new_bond


class MoleculeGraph:
    '''
    Graph data structure for handle molecule
    '''

    # ------------------ init --------------------------- #

    def __init__(self,
                 mol: Molecule = None,
                 coords_conf_id: int = None,
                 aromaticity: str = "rdkit",
                 max_include_ring: int = 8):
        '''
        Parameters
        ----------
        mol : core.Molecule
        bonds : list[(bond_begin, bond_end)]
        coords_conf_id: if set coords_conf_id, save coords from rkmol.conformer with conf_id 
        '''
        self._atom_by_idx = dict()
        self._bond_by_idx = dict()
        self.topo = None
        self.has_coord = False
        self.rkmol = None
        self.aromaticity = aromaticity
        self.max_include_ring = max_include_ring

        if mol is not None:
            assert isinstance(mol, Molecule)
            assert "." not in mol.get_smiles()
            self.rkmol = rkutil.sanitize_rkmol(mol.get_rkmol(), aromaticity=self.aromaticity)
            self._create_from_rkmol()
            if coords_conf_id is not None:
                coords = mol.get_conformer(coords_conf_id).coords
                self._set_coord(coords)
        if self._atom_by_idx:
            self.update_graph_info()

    def _create_from_rkmol(self):
        '''
        add the whole molecular graph
        '''
        bonds = list()
        bond_ids = list()
        for atom in self.rkmol.GetAtoms():
            new_atom = Atom(atom)
            self._atom_by_idx[new_atom.idx] = new_atom
        for bond in self.rkmol.GetBonds():
            new_bond = Bond(bond)
            self._bond_by_idx[new_bond.idx] = new_bond
            bonds.append((new_bond.begin_idx, new_bond.end_idx))
            bond_ids.append(new_bond.idx)
        self.topo = Topology(bonds=bonds, bond_ids=bond_ids, max_include_ring=self.max_include_ring)

    def _set_coord(self, coords: np.ndarray):
        assert coords.shape == (self.natoms, 3)
        for atom in self.get_atoms():
            atom.coord = coords[atom.idx]
        self.has_coord = True

    def update_graph_info(self, rings: list[list[int]] = None):
        if rings is None:
            rings = self.get_rings()
        # clear stored info
        for bond in self.get_graph_bonds():
            bond.in_ring = False
        # set bond ring info
        for ring in rings:
            for pair in zip(ring, ring[1:] + [ring[0]]):
                bond = self.get_connect_bond(*[self.get_atom(idx) for idx in pair])
                bond.in_ring = True
        # set atom ring info
        for atom in self.get_atoms():
            atom.connectivity = len(self.get_neighbors(atom))
            atom.ring_connectivity = len([bond for bond in self.get_neighbor_bonds(atom) if bond.in_ring])
            ring_sizes = [len(ring) for ring in rings if atom.idx in ring]
            atom.min_ring_size = min(ring_sizes) if ring_sizes else 0

    # ------------------ get informations --------------- #

    @property
    def natoms(self) -> int:
        return self.topo.natoms

    @property
    def nbonds(self) -> int:
        return self.topo.nbonds

    def get_atom(self, idx: int) -> Atom:
        return self._atom_by_idx[idx]

    def get_bond_by_idx(self, idx: int) -> Bond:
        return self._bond_by_idx[idx]

    def get_bond(self, atom1: int, atom2: int) -> Bond:
        bond = self.topo.graph.get_edge_data(atom1, atom2)
        if bond is not None:
            return self.get_bond_by_idx(bond['idx'])
        else:
            return None

    def get_connect_bond(self, atom1: Atom, atom2: Atom) -> Bond:
        ''' bond between the two given atoms or None if not connected
        '''
        bond = self.topo.graph.get_edge_data(atom1.idx, atom2.idx)
        if bond is not None:
            return self._bond_by_idx[bond['idx']]
        return None

    def get_atoms(self) -> list[Atom]:
        ''' all atoms
        '''
        return list(self._atom_by_idx.values())

    def get_sorted_atoms(self) -> list[Atom]:
        ''' all atoms sorted by idx
        '''
        atoms = self.get_atoms()
        atoms.sort(key=attrgetter("idx"))
        return atoms

    def get_graph_bonds(self) -> list[Bond]:
        ''' all bonds
        '''
        return list(self._bond_by_idx.values())

    def get_bond_with_atoms(self) -> list[tuple[Bond, Atom, Atom]]:
        return [(self._bond_by_idx[data['idx']], self._atom_by_idx[a1], self._atom_by_idx[a2])
                for a1, a2, data in self.topo.graph.edges(data=True)]

    def get_neighbors(self, atom: Atom) -> list[Atom]:
        ''' list of atoms one bond (edge) away from the given atom
        '''
        return [self._atom_by_idx[idx] for idx in self.topo.adj_list[atom.idx]]

    def get_neighbor_ids(self, atom_id: int) -> list[int]:
        ''' get neighbor index by atom idx
        '''
        return self.topo.adj_list[atom_id].copy()

    def get_neighbor_bonds(self, atom: Atom) -> list[Bond]:
        return [self.get_connect_bond(atom, nei) for nei in self.get_neighbors(atom)]

    def get_bonds(self) -> list[tuple[int]]:
        return self.topo.bonds

    def get_angles(self) -> list[tuple[int]]:
        ''' return list of angles represented by atom ids
        '''
        return self.topo.angles

    def get_propers(self) -> list[tuple[int]]:
        ''' return list of propers represented by atom ids
        '''
        return self.topo.propers

    def get_atoms_with_three_neighbors(self) -> list[tuple[int]]:
        ''' return list of central atoms with three neighbors represented by atom ids
        '''
        return self.topo.atoms_with_three_neighbors

    def get_atoms_with_one_neighbor(self) -> list[tuple[int]]:
        ''' return list of central atoms with one neighbor represented by atom ids
        '''
        return self.topo.atoms_with_one_neighbor

    def get_impropers(self) -> list[tuple[int]]:
        atomsets = set()
        for atom in self.get_atoms():
            if atom.atomic_number in [6, 7]:
                idx = atom.idx
                nei = self.get_neighbor_ids(idx)
                if len(nei) == 3:
                    atomsets.add(rkutil.sorted_atomids((idx, nei[0], nei[1], nei[2]), is_improper=True))
        atomsets = list(atomsets)
        atomsets.sort(key=itemgetter(0, 1, 2, 3))
        return atomsets

    def get_nonbonded_pairs(self) -> tuple[list[tuple], list[tuple]]:
        return self.topo.nonbonded14_pairs, self.topo.nonbondedall_pairs

    def get_nonbonded14_pairs(self) -> tuple[list[tuple], list[tuple]]:
        return self.topo.nonbonded14_pairs

    def get_nonring_dihedral_rotate_atoms(self, *indices: list[int]) -> list[int]:
        return self.topo.get_nonring_dihedral_rotate_atoms(*indices)

    def get_intra_topo(self) -> dict[str, list[tuple[int]]]:
        '''
        get all bonds, angles and propers topology in the molecule
        '''
        bonds = self.get_bonds()
        angles = self.get_angles()
        propers = self.get_propers()
        impropers = self.get_impropers()
        return {'Bond': bonds, 'Angle': angles, 'ProperTorsion': propers, 'ImproperTorsion': impropers}

    def get_tfd_propers(self) -> list[tuple[int]]:
        '''get tfd rotatable propers'''
        rkmol = self.get_rkmol()
        return rkutil.get_tfd_propers(rkmol)

    def get_rkmol(self) -> Chem.Mol:
        return Chem.Mol(self.rkmol)

    def get_rings(self) -> list[list[int]]:
        return self.topo.rings

    def get_aromatic_rings(self, max_size=7) -> list[list[int]]:

        def check_aromaticity(_ring):
            n_ab = 0
            for i in range(len(_ring)):
                bond = self.get_bond(_ring[i - 1], _ring[i])
                if 1.4 < bond.order < 1.6:
                    n_ab += 1
            if n_ab >= len(_ring) - 1:  # handle Azulene
                return True
            else:
                return False

        rings = [list(l) for l in Chem.GetSymmSSSR(self.rkmol)]
        arings = []
        for ring in rings:
            if len(ring) <= max_size:
                if check_aromaticity(ring):
                    arings.append(ring)
        return arings

    def get_linear_propers(self) -> list[tuple[int]]:
        ''' Return all linear propers '''
        rkmol = self.get_rkmol()
        linear_patternC = Chem.MolFromSmarts('[*:1]~[#6X2;!r5;!r6:2]~[*:3]~[*:4]')
        linear_patternN = Chem.MolFromSmarts('[!$([#7H1]):1]~[#7X2+:2]~[!$([#7H1]):3]~[*:4]')
        matchesC = find_mapped_smarts_matches(rkmol, linear_patternC)
        matchesN = find_mapped_smarts_matches(rkmol, linear_patternN)
        matches = matchesC.union(matchesN)
        atomsets = set([rkutil.sorted_atomids(proper) for proper in matches])
        linear_propers = list(atomsets)
        linear_propers.sort(key=itemgetter(0, 1, 2, 3))
        return linear_propers

    def _calc_priority(self, i: int, j: int, k: int, l: int, in_ring: bool) -> tuple:
        ''' Select one among all propers corresponding to bond (j,k).
        Calculate priority of proper [i,j,k,l] and return a tuple including
        ring_order: 5 if j,k are not in the same ring,
                    4 if i,j,k,l in the same ring,
                    3 if i,j,k in the same ring,
                    2 if j,k,l in the same ring,
                    1 if j,k in the same ring;
        bond_order: summation of bond orders of bond (i,j) and (k,l);
        connectivity: summation of connections of atom i and atom l;
        atomic_number: summation of atomic numbers of atom i and atom l.
        The largest one will be selected.
        '''
        bond_ij = self.get_bond(i, j)
        bond_kl = self.get_bond(k, l)
        bond_order = bond_ij.order * bond_kl.order
        degree = len(self.get_neighbor_ids(i)) + len(self.get_neighbor_ids(l))
        atom_i = self.get_atom(i)
        atom_l = self.get_atom(l)
        atomic_number = atom_i.atomic_number + atom_l.atomic_number
        ring_order = 0
        if in_ring:  #j,k in ring
            for ring in self.get_rings():
                if j in ring and k in ring:
                    ring_order = max(1, ring_order)  #j,k in the same ring
                if i in ring and l in ring:
                    ring_order = 4  # i,j,k,l in the same ring
                    break
                elif i in ring:
                    ring_order = max(3, ring_order)  # i,j,k in the same ring
                elif l in ring:
                    ring_order = max(2, ring_order)  # j,k,l in the same ring
        else:
            ring_order = 5  # j,k not in ring
        if ring_order == 0:
            ring_order = 5  # j,k in ring but not in the same ring
        return tuple([ring_order, bond_order, degree, atomic_number])

    def get_proper_from_bonds(self, bonds: list[tuple[int, int]]) -> list[tuple[int]]:
        ''' Given a list of bonds, return a proper list.
        Only one proper is selected for each bond. 
        The priority of proper is calculated by function _calc_priority
        '''
        atomsets = set()
        for atom_ids in bonds:
            j = atom_ids[0]
            k = atom_ids[1]
            bond = self.get_bond(j, k)
            if bond == None:
                continue
            list_i = self.get_neighbor_ids(j)
            list_i.remove(k)
            list_l = self.get_neighbor_ids(k)
            list_l.remove(j)
            priority_max = (0, 0, 0, 0)
            proper = list()
            for i in list_i:
                for l in list_l:
                    priority = self._calc_priority(i, j, k, l, bond.in_ring)
                    if priority > priority_max:
                        priority_max = priority
                        proper = [i, j, k, l]
            atomsets.add(rkutil.sorted_atomids(proper))
        propers = list(atomsets)
        propers.sort(key=itemgetter(0, 1, 2, 3))
        return propers

    def get_nonring_rotatable_bonds(self) -> list[tuple[int, int]]:
        ''' return nonring rotatable bonds '''
        rkmol = self.get_rkmol()
        nonring_pattern = Chem.MolFromSmarts('[!D1]-,=;!@[!D1]')
        matches = find_mapped_smarts_matches(rkmol, nonring_pattern)
        atomsets = set([rkutil.sorted_atomids(bond) for bond in matches])
        bonds = list(atomsets)
        bonds.sort(key=itemgetter(0, 1))
        return bonds

    def get_nonring_rotatable_propers(self) -> list[tuple[int]]:
        ''' return nonring rotatable propers and linear propers are excluded '''
        linear = self.get_linear_propers()
        bonds = self.get_nonring_rotatable_bonds()
        propers = self.get_proper_from_bonds(bonds)
        propers = [proper for proper in propers if not proper in linear]
        return propers

    def get_ring_rotatable_bonds(self) -> list[tuple[int, int]]:
        ''' return ring rotatable bonds '''
        rkmol = self.get_rkmol()
        ring_pattern = Chem.MolFromSmarts('[!D1]-,=;@[!D1]')
        matches = find_mapped_smarts_matches(rkmol, ring_pattern)
        small_ring = set()
        for bond in matches:
            for ring in self.get_rings():
                if bond[0] in ring and bond[1] in ring and len(ring) < 4:
                    small_ring.add(bond)
                    break
        matches = matches - small_ring
        atomsets = set([rkutil.sorted_atomids(bond) for bond in matches])
        bonds = list(atomsets)
        bonds.sort(key=itemgetter(0, 1))
        return bonds

    def get_ring_rotatable_propers(self) -> list[tuple[int]]:
        ''' return ring rotatable propers and linear propers are excluded '''
        linear = self.get_linear_propers()
        bonds = self.get_ring_rotatable_bonds()
        propers = self.get_proper_from_bonds(bonds)
        propers = [proper for proper in propers if not proper in linear]
        return propers

    def get_rotatable_bonds(self) -> list[tuple[int, int]]:
        ''' return all rotatable bonds '''
        bonds_ring = self.get_ring_rotatable_bonds()
        bonds_nonring = self.get_nonring_rotatable_bonds()
        bonds = bonds_ring + bonds_nonring
        bonds.sort(key=itemgetter(0, 1))
        return bonds

    def get_rotatable_propers(self) -> list[tuple[int]]:
        ''' return all rotatable propers and linear propers are excluded '''
        linear = self.get_linear_propers()
        bonds = self.get_rotatable_bonds()
        propers = self.get_proper_from_bonds(bonds)
        propers = [proper for proper in propers if not proper in linear]
        return propers


class MutableMoleculeGraph(MoleculeGraph):
    # ---------------- output graph ------------------------- #

    def to_rkmol(self, set_coord=False):
        '''
        convert MutableMoleculeGraph to rdkit.Chem.Mol
        '''
        mol = Chem.RWMol()
        # map index in molgraph to index in rkmol
        new_ids = dict()
        # map index in rkmol to index in molgraph
        old_ids = dict()
        for idx, atom in self._atom_by_idx.items():
            assert atom.atomic_number is not None, "No element information!"
            # create atom
            rdatom = Chem.Atom(rkutil.atomnum_elem[atom.atomic_number])
            # set charge
            rdatom.SetFormalCharge(round(atom.formal_charge))
            # add atom
            new_idx = mol.AddAtom(rdatom)
            new_ids[idx] = new_idx
            old_ids[new_idx] = idx
        for bond, a1, a2 in self.get_bond_with_atoms():
            bond_indices = [new_ids[a1.idx], new_ids[a2.idx]]
            bond_type = rkutil.num_bondorder.get(bond.order, Chem.BondType.SINGLE)
            mol.AddBond(*bond_indices, bond_type)
        if set_coord:
            assert self.has_coord
            # assign coordinates
            conf = Chem.Conformer(mol.GetNumAtoms())
            for atom in mol.GetAtoms():
                idx = atom.GetIdx()
                xyz = self._atom_by_idx[old_ids[idx]].coord
                conf.SetAtomPosition(idx, xyz)
            mol.AddConformer(conf)
            mol.UpdatePropertyCache(strict=False)
            # assign R/S to atoms and Z/E to bonds
            # reference: https://github.com/rdkit/rdkit/blob/bc8d2e7556d33ecbb181890e9050d60b5ac75b7a/Code/GraphMol/DetermineBonds/DetermineBonds.cpp#L289
            Chem.SetDoubleBondNeighborDirections(mol)
            Chem.AssignStereochemistryFrom3D(mol)
        mol = rkutil.sanitize_rkmol(mol, aromaticity=self.aromaticity)
        return mol

    def to_molecule(self, set_coord=False) -> Molecule:
        rkmol = self.to_rkmol(set_coord=set_coord)
        mol = Molecule.from_rdkit(rkmol)
        return mol

    # ---------------- edit graph --------------------------- #
    def update_topo(self):
        bonds = list()
        bond_ids = list()
        for bond_idx in self._bond_by_idx.keys():
            bond_ids.append(bond_idx)
            bonds.append((self._bond_by_idx[bond_idx].begin_idx, self._bond_by_idx[bond_idx].end_idx))
        self.topo = Topology(bonds=bonds, bond_ids=bond_ids)

    def remove_atom(self, atom: Atom, update: bool = True):
        bonds = self.get_neighbor_bonds(atom)
        for bond in bonds:
            self._bond_by_idx.pop(bond.idx)
        self._atom_by_idx.pop(atom.idx)
        if update:
            self.update_topo()

    def add_atom(self, atom: Atom, update: bool = True):
        assert isinstance(atom, Atom)
        assert atom.idx not in self._atom_by_idx
        self._atom_by_idx[atom.idx] = atom
        if update:
            self.update_topo()

    def add_bond(self, bond: Bond, atom1: Atom, atom2: Atom, update: bool = True):
        assert isinstance(bond, Bond)
        assert isinstance(atom1, Atom)
        assert isinstance(atom2, Atom)
        assert bond.idx not in self._bond_by_idx
        assert atom1.idx in self._atom_by_idx
        assert atom2.idx in self._atom_by_idx
        bond.begin_idx = atom1.idx
        bond.end_idx = atom2.idx
        self._bond_by_idx[bond.idx] = bond
        if update:
            self.update_topo()
