def extract_residue_accessibility(in_rsa, path=True, get_total=False):
    """Parses rsa file for solvent accessibility for each residue.

    Parameters
    ----------
    in_rsa : str
        Path to naccess rsa file
    path : bool
        Indicates if in_rsa is a path or a string
    get_total : bool
        Indicates if the total accessibility from the file needs to
        be extracted. Convenience method for running the
        total_accessibility function but only running NACCESS once

    Returns
    -------
    rel_solv_ac_acc_atoms : list
        Relative solvent accessibility of all atoms in each amino acid
    get_total : float
        Relative solvent accessibility of all atoms in the NACCESS rsa file
    """

    if path:
        with open(in_rsa, 'r') as inf:
            rsa = inf.read()
    else:
        rsa = in_rsa[:]

    residue_list = [x for x in rsa.splitlines()]
    rel_solv_acc_all_atoms = [
        float(x[22:28])
        for x in residue_list
        if x[0:3] == "RES" or x[0:3] == "HEM"]

    if get_total:
        (all_atoms, _, _, _, _) = total_accessibility(
            rsa, path=False)
        return rel_solv_acc_all_atoms, all_atoms
    return rel_solv_acc_all_atoms, None