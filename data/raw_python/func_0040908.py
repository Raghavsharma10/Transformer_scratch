def dset_info(dset):
    '''returns a :class:`DsetInfo` object containing the meta-data from ``dset``'''
    if nl.pkg_available('afni'):
        return _dset_info_afni(dset)
    nl.notify('Error: no packages available to get dset info',level=nl.level.error)
    return None