def smilin(smiles, transforms=[figueras.sssr, aromaticity.aromatize]):
    """(smiles)->molecule
    Convert a smiles string into a molecule representation"""
    builder = BuildMol()
    tokenize(smiles, builder)
    mol = builder.mol

        
    for transform in transforms:
        mol = transform(mol)

    ## implicit hcount doesn't make any sense anymore...
    for atom in mol.atoms:
        if not atom.has_explicit_hcount:
            atom.imp_hcount = atom.hcount - atom.explicit_hcount

    return mol