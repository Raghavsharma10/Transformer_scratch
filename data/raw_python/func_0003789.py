def compute_rotsym(molecule, graph, threshold=1e-3*angstrom):
    """Compute the rotational symmetry number

       Arguments:
        | ``molecule``  --  The molecule
        | ``graph``  --  The corresponding bond graph

       Optional argument:
        | ``threshold``  --  only when a rotation results in an rmsd below the
                             given threshold, the rotation is considered to
                             transform the molecule onto itself.
    """
    result = 0
    for match in graph.symmetries:
        permutation = list(j for i,j in sorted(match.forward.items()))
        new_coordinates = molecule.coordinates[permutation]
        rmsd = fit_rmsd(molecule.coordinates, new_coordinates)[2]
        if rmsd < threshold:
            result += 1
    return result