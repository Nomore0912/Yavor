#!/usr/bin/env python
# -*- coding:utf-8 -*-
from Bio.PDB import PDBParser


def load_pdb(file_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', file_path)
    return structure


def get_residues(structure, chain_id, start, end):
    chain = structure[chain_id]
    residues = [res for res in chain if start <= res.get_id()[1] <= end]
    return residues


