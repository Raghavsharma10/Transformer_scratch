def value_at_coord(dset,coords):
    '''returns value at specified coordinate in ``dset``'''
    return nl.numberize(nl.run(['3dmaskave','-q','-dbox'] + list(coords) + [dset],stderr=None).output)