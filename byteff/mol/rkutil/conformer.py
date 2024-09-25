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

import logging
from typing import List, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry.rdGeometry import Point3D

logger = logging.getLogger(__name__)

##########################################
##           conformers
##########################################


def opt_confs(mol: Chem.Mol, *, ffoptimizer: str = 'mmff94s', n_threads: int = 1, verbose: bool = False) -> List:
    """optimization conformers and calculated energies.

    Args:
    mol: rdkit mol
    ffoptimizer: the force field used to optimize conformers.
    n_threads: `0` means use all threads.
    is_verbose: used to debug.

    Returns:
    energies: the force field opt energies.
    """

    ffoptimizer = ffoptimizer.lower()
    ffoptimizers = ["mmff94", "mmff94s", "uff"]
    if ffoptimizer not in ffoptimizers:
        raise NotImplementedError(f"{ffoptimizer} is not avaliable, choose one from {ffoptimizers}")

    results = None
    ffopt_max_cycles = 20000

    n_confs = mol.GetNumConformers()
    if verbose:
        logger.info(f"There are {n_confs} conformers.")

    if ffoptimizer in ["mmff94", "mmff94s"]:
        results = AllChem.MMFFOptimizeMoleculeConfs(mol,
                                                    mmffVariant=ffoptimizer,
                                                    numThreads=n_threads,
                                                    maxIters=ffopt_max_cycles,
                                                    ignoreInterfragInteractions=False)

    elif ffoptimizer == "uff":
        results = AllChem.UFFOptimizeMoleculeConfs(mol,
                                                   numThreads=n_threads,
                                                   maxIters=ffopt_max_cycles,
                                                   ignoreInterfragInteractions=False)

    # gather energies in hartree and check results
    # results is a list of (not_converged, energy) 2-tuples. energy in kcal/mol
    # If not_converged is 0 the optimization converged for that conformer.
    assert results is not None, f'OptimizeMoleculeConfs failed with {ffoptimizer}'

    return results


def generate_confs(rkmol: Chem.Mol,
                   *,
                   nconfs: int = 1,
                   ffopt: bool = True,
                   n_threads: int = 1,
                   verbose: bool = False,
                   **kwargs) -> Tuple:
    """ generate conformers for rkmol

    REQUIREMENT: rkmol must have been sanitized, otherwise an error is raised

    Args:
    rkmol:
    nconfs: The target number of conformers to generate.

    Returns:
    elements: the elements of atoms in mol.
    coordinates: all conformers coordination.
    energies: opt energies.
    mol:

    Raises:
    ValueError:
        1. Embed molecules failed.
        2. Opt failed.
    """

    # embed conformers settings
    # https://www.rdkit.org/docs/RDKit_Book.html?highlight=etversion#parameters-controlling-conformer-generation
    params = AllChem.EmbedParameters()
    params.ETversion = 2  # for both ETKDGv2 and ETKDGv3 this should be 2
    params.useSymmetryForPruning = kwargs.pop('useSymmetryForPruning', True)
    params.useBasicKnowledge = kwargs.pop('useBasicKnowledge', True)
    params.enforceChirality = kwargs.pop('enforceChirality', True)

    params.useSmallRingTorsions = kwargs.pop('useSmallRingTorsions', False)
    params.useMacrocycleTorsions = kwargs.pop('useMacrocycleTorsions', False)
    params.useExpTorsionAnglePrefs = kwargs.pop('useExpTorsionAnglePrefs', False)

    params.useRandomCoords = kwargs.pop('useRandomCoords', False)  # useful for large molecules
    params.randomSeed = kwargs.pop('randomSeed', 42)
    params.pruneRmsThresh = kwargs.pop('pruneRmsThresh', -1.0)  # no prune by default
    params.maxIterations = kwargs.pop('maxIterations', 1000)
    params.numThreads = n_threads

    params.clearConfs = True
    params.embedFragmentsSeparately = True
    params.forceTransAmides = True

    n_confs = nconfs * kwargs.pop('nconfs_multiplier', 1)

    conf_ids = AllChem.EmbedMultipleConfs(rkmol, n_confs, params)
    if not conf_ids and not params.useRandomCoords:
        params.useRandomCoords = True
        if verbose:
            logger.info("Retry EmbedMultipleConfs using random_coord=True")
        conf_ids = AllChem.EmbedMultipleConfs(rkmol, n_confs, params)

    if not conf_ids:
        raise ValueError("EmbedMultipleConfs failed.")
    if verbose:
        logger.info(f"{len(conf_ids)}/{n_confs} conformers are generated.")

    if ffopt:
        results = opt_confs(rkmol, ffoptimizer='mmff94s', n_threads=n_threads, verbose=verbose)
        success = [1 if r[0] == 0 else 0 for r in results]
        energy_kcal = [r[1] for r in results]

        if verbose:
            logger.info(f"{sum(success)}/{n_confs} conformers are optimized.")
            logger.info(f"energies: {energy_kcal}")

        return rkmol, success, energy_kcal
    else:
        return rkmol, None, None


def append_conformers_to_mol(mol: Chem.Mol, conformers: List[np.ndarray]) -> Chem.Mol:
    '''append conformers to rkmol. existing conformers in rkmol are unchanged'''
    rkmol = Chem.Mol(mol)
    natoms = rkmol.GetNumAtoms(onlyExplicit=False)
    for coords in conformers:
        assert coords.shape == (natoms, 3)
        rkconf = Chem.Conformer(natoms)
        for atom_idx in range(natoms):
            atom_pos = Point3D(*coords[atom_idx].tolist())
            rkconf.SetAtomPosition(atom_idx, atom_pos)
        rkmol.AddConformer(rkconf, assignId=True)
    return rkmol
