def write_image_dataset(group, key, data, h5dtype=None):
    """Write an image to an hdf5 group as a dataset

    This convenience function sets all attributes such that the image
    can be visualized with HDFView, sets the compression and fletcher32
    filters, and sets the chunk size to the image shape.

    Parameters
    ----------
    group: h5py.Group
        HDF5 group to store data to
    key: str
        Dataset identifier
    data: np.ndarray of shape (M,N)
        Image data to store
    h5dtype: str
        The datatype in which to store the image data. The default
        is the datatype of `data`.

    Returns
    -------
    dataset: h5py.Dataset
        The created HDF5 dataset object
    """
    if h5dtype is None:
        h5dtype = data.dtype
    if key in group:
        del group[key]
    if group.file.driver == "core":
        kwargs = {}
    else:
        kwargs = {"fletcher32": True,
                  "chunks": data.shape}
        kwargs.update(COMPRESSION)

    dset = group.create_dataset(key,
                                data=data.astype(h5dtype),
                                **kwargs)
    # Create and Set image attributes
    # HDFView recognizes this as a series of images
    dset.attrs.create('CLASS', b'IMAGE')
    dset.attrs.create('IMAGE_VERSION', b'1.2')
    dset.attrs.create('IMAGE_SUBCLASS', b'IMAGE_GRAYSCALE')
    return dset