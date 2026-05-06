def dump_dict_of_nested_lists_to_h5(fname, data):
    """
    Take nested list structure and dump it in hdf5 file.


    Parameters
    ----------
    fname : str 
        Filename
    data : dict(list(numpy.ndarray))
        Dict of nested lists with variable len arrays.
    
    
    Returns
    -------
    None

    """
    # Open file
    print('writing to file: %s' % fname)
    f = h5py.File(fname)
    # Iterate over values
    for i, ivalue in list(data.items()):
        igrp = f.create_group(str(i))
        for j, jvalue in enumerate(ivalue):
            jgrp = igrp.create_group(str(j))
            for k, kvalue in enumerate(jvalue):
                if kvalue.size > 0:
                    dset = jgrp.create_dataset(str(k), data=kvalue,
                                               compression='gzip')
                else:
                    dset = jgrp.create_dataset(str(k), data=kvalue,
                                               maxshape=(None, ),
                                               compression='gzip')
    # Close file
    f.close()