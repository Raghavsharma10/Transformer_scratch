def find_ss_regions(dssp_residues, loop_assignments=(' ', 'B', 'S', 'T')):
    """Separates parsed DSSP data into groups of secondary structure.

    Notes
    -----
    Example: all residues in a single helix/loop/strand will be gathered
    into a list, then the next secondary structure element will be
    gathered into a separate list, and so on.

    Parameters
    ----------
    dssp_residues : [tuple]
        Each internal list contains:
            [0] int Residue number
            [1] str Secondary structure type
            [2] str Chain identifier
            [3] str Residue type
            [4] float Phi torsion angle
            [5] float Psi torsion angle
            [6] int dssp solvent accessibility

    Returns
    -------
    fragments : [[list]]
        Lists grouped in continuous regions of secondary structure.
        Innermost list has the same format as above.
    """

    loops = loop_assignments
    previous_ele = None
    fragment = []
    fragments = []
    for ele in dssp_residues:
        if previous_ele is None:
            fragment.append(ele)
        elif ele[2] != previous_ele[2]:
            fragments.append(fragment)
            fragment = [ele]
        elif previous_ele[1] in loops:
            if ele[1] in loops:
                fragment.append(ele)
            else:
                fragments.append(fragment)
                fragment = [ele]
        else:
            if ele[1] == previous_ele[1]:
                fragment.append(ele)
            else:
                fragments.append(fragment)
                fragment = [ele]
        previous_ele = ele
    fragments.append(fragment)
    return fragments