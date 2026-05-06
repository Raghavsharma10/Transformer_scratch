def extract_all_ss_dssp(in_dssp, path=True):
    """Uses DSSP to extract secondary structure information on every residue.

    Parameters
    ----------
    in_dssp : str
        Path to DSSP file.
    path : bool, optional
        Indicates if pdb is a path or a string.

    Returns
    -------
    dssp_residues : [tuple]
        Each internal list contains:
            [0] int Residue number
            [1] str Secondary structure type
            [2] str Chain identifier
            [3] str Residue type
            [4] float Phi torsion angle
            [5] float Psi torsion angle
            [6] int dssp solvent accessibility
    """

    if path:
        with open(in_dssp, 'r') as inf:
            dssp_out = inf.read()
    else:
        dssp_out = in_dssp[:]
    dssp_residues = []
    active = False
    for line in dssp_out.splitlines():
        if active:
            try:
                res_num = int(line[5:10].strip())
                chain = line[10:12].strip()
                residue = line[13]
                ss_type = line[16]
                phi = float(line[103:109].strip())
                psi = float(line[109:116].strip())
                acc = int(line[35:38].strip())
                dssp_residues.append(
                    (res_num, ss_type, chain, residue, phi, psi, acc))
            except ValueError:
                pass
        else:
            if line[2] == '#':
                active = True
    return dssp_residues