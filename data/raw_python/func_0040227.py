def max_diff(dset1,dset2):
    '''calculates maximal voxel-wise difference in datasets (in %)

    Useful for checking if datasets have the same data. For example, if the maximum difference is
    < 1.0%, they're probably the same dataset'''
    for dset in [dset1,dset2]:
        if not os.path.exists(dset):
            nl.notify('Error: Could not find file: %s' % dset,level=nl.level.error)
            return float('inf')
    try:
        dset1_d = nib.load(dset1)
        dset2_d = nib.load(dset2)
        dset1_data = dset1_d.get_data()
        dset2_data = dset2_d.get_data()
    except IOError:
        nl.notify('Error: Could not read files %s and %s' % (dset1,dset2),level=nl.level.error)
        return float('inf')
    try:
        old_err = np.seterr(divide='ignore',invalid='ignore')
        max_val = 100*np.max(np.ma.masked_invalid(np.double(dset1_data - dset2_data) / ((dset1_data+dset2_data)/2)))
        np.seterr(**old_err)
        return max_val
    except ValueError:
        return float('inf')