# BDTorsion Benchmark Data

This folder contains data for BDTorsion.
The two subsets (InRing and NonRing) are saved seperated in JSON and H5 files, respectively.

The JSON files contain the mapping between molecule names and mapped SMILES. 
Mapped SMILES can be parsed by RDKit to reconstruct molecular graphs.

The data in the H5 files are grouped by molecule names.
Within each group, there are three datasets: `coords`, `forces` and `energy`. 
The shape of `coords` and `forces` is [# conformers, # atoms, 3], and the shape of `energy` is [# conformers].