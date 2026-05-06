def copyh5(inh5, outh5):
    """Recursively copy all hdf5 data from one group to another

    Data from links is copied.

    Parameters
    ----------
    inh5: str, h5py.File, or h5py.Group
        The input hdf5 data. This can be either a file name or
        an hdf5 object.
    outh5: str, h5py.File, h5py.Group, or None
        The output hdf5 data. This can be either a file name or
        an hdf5 object. If set to `None`, a new hdf5 object is
        created in memory.

    Notes
    -----
    All data in outh5 are overridden by the inh5 data.
    """
    if not isinstance(inh5, h5py.Group):
        inh5 = h5py.File(inh5, mode="r")
    if outh5 is None:
        # create file in memory
        h5kwargs = {"name": "qpimage{}.h5".format(QPImage._instances),
                    "driver": "core",
                    "backing_store": False,
                    "mode": "a"}
        outh5 = h5py.File(**h5kwargs)
        return_h5obj = True
        QPImage._instances += 1
    elif not isinstance(outh5, h5py.Group):
        # create new file
        outh5 = h5py.File(outh5, mode="w")
        return_h5obj = False
    else:
        return_h5obj = True
    # begin iteration
    for key in inh5:
        if key in outh5:
            del outh5[key]
        if isinstance(inh5[key], h5py.Group):
            outh5.create_group(key)
            copyh5(inh5[key], outh5[key])
        else:
            dset = write_image_dataset(group=outh5,
                                       key=key,
                                       data=inh5[key][:],
                                       h5dtype=inh5[key].dtype)
            dset.attrs.update(inh5[key].attrs)
    outh5.attrs.update(inh5.attrs)
    if return_h5obj:
        # in-memory or previously created instance of h5py.File
        return outh5
    else:
        # properly close the file and return its name
        fn = outh5.filename
        outh5.flush()
        outh5.close()
        return fn