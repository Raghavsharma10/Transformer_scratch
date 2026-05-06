def skull_strip(dset,suffix='_ns',prefix=None,unifize=True):
    '''attempts to cleanly remove skull from ``dset``'''
    return available_method('skull_strip')(dset,suffix,prefix,unifize)