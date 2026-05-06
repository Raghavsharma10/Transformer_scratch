def fix_even_row_data_fc(fdata):
    """When the number of rows in fdata is even, there is a subtlety that must
    be taken care of if fdata is to satisfy the symmetry required for further
    processing. For an array length of 6,the data is align as [0 1 2 -3 -2 -1] 
    this routine simply sets the row corresponding to the -3 index equal to 
    zero. It is an unfortunate subtlety, but not taking care of this has
    resulted in answers that are not double precision. This operation should
    be applied before any other operators are applied to fdata."""

    L = fdata.shape[0]
    if np.mod(L, 2) == 0:
        fdata[int(L / 2), :] = 0