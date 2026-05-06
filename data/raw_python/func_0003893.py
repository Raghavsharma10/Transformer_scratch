def fit_rmsd(ras, rbs, weights=None):
    """Fit geometry rbs onto ras, returns more info than superpose

       Arguments:
        | ``ras``  --  a numpy array with 3D coordinates of geometry A,
                       shape=(N,3)
        | ``rbs``  --  a numpy array with 3D coordinates of geometry B,
                       shape=(N,3)

       Optional arguments:
        | ``weights``  --  a numpy array with fitting weights for each
                           coordinate, shape=(N,)

       Return values:
        | ``transformation``  --  the transformation that brings geometry A into
                                  overlap with geometry B
        | ``rbs_trans``  --  the transformed coordinates of geometry B
        | ``rmsd``  --  the rmsd of the distances between corresponding atoms in
                        geometry A and B

       This is a utility routine based on the function superpose. It just
       computes rbs_trans and rmsd after calling superpose with the same
       arguments
    """
    transformation = superpose(ras, rbs, weights)
    rbs_trans = transformation * rbs
    rmsd = compute_rmsd(ras, rbs_trans)
    return transformation, rbs_trans, rmsd