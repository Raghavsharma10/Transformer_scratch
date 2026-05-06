def blur(dset,fwhm,prefix=None):
    '''blurs ``dset`` with given ``fwhm`` runs 3dmerge to blur dataset to given ``fwhm``
    default ``prefix`` is to suffix ``dset`` with ``_blur%.1fmm``'''
    if prefix==None:
        prefix = nl.suffix(dset,'_blur%.1fmm'%fwhm)
    return available_method('blur')(dset,fwhm,prefix)