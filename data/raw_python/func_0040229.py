def create_dset(directory,slice_order='alt+z',sort_order='zt',force_slices=None):
    '''tries to autocreate a dataset from images in the given directory'''
    return _create_dset_dicom(directory,slice_order,sort_order,force_slices=force_slices)