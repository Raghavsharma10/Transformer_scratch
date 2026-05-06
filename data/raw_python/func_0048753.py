def unwrapArray(a, recursive=True, readH5pyDataset=True):
    """ This function takes an object (like a dictionary) and recursively
        unwraps it solving issues like:
          * the fact that many objects are packaged as 0d array
        This funciton has also some specific hack for handling h5py limits:
          * handle the None python object
          * numpy unicode ...
    """
    try:

        ### take care of hdf5 groups 
        if isinstance(a,h5py.Group):
            # take care of special flags first
            if isinstance(a, h5py.Group) and ( ("IS_LIST" in a.attrs) or ("IS_LIST_OF_ARRAYS" in a.attrs) ):
                items = list(a.keys())
                items.sort()
                a = [unwrapArray(a[item],readH5pyDataset=readH5pyDataset) for item in items]


        ### take care of hdf5 datasets
        elif isinstance(a,h5py.Dataset):

            # read if asked so or if dummy array
            # WARNING: a.value and a[...] do not return the
            # same thing... 
            # a[...] returns ndarray if a is a string
            # a.value returns a str(py3) or unicode(py2)
            if readH5pyDataset or a.shape == (): a = a.value#[...]


        # special None flag
        # not array needed for FutureWarning: elementwise comparison failed; ...
        if not isinstance(a,np.ndarray) and a == "NONE_PYTHON_OBJECT": a = None
 
        # clean up non-hdf5 specific
        if isinstance(a, np.ndarray) and a.ndim == 0:
            a = a.item()

        # convert to str (for example h5py can't save numpy unicode)
        if isinstance(a, np.ndarray) and a.dtype.char == "S":
            a = a.astype(str)

        if recursive:
            if "items" in dir(a):  # dict, h5py groups, npz file
                a = dict(a)  # convert to dict, otherwise can't asssign values
                for key, value in a.items():
                    a[key] = unwrapArray(value,readH5pyDataset=readH5pyDataset)
            elif isinstance(a, (list, tuple)):
                a = [unwrapArray(element,readH5pyDataset=readH5pyDataset) 
                    for element in a]
            else:
                pass

    except Exception as e:
        log.warning("Could not handle %s, error was: %s"%(a,str(e)))
    return a