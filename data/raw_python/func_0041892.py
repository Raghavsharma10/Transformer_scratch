def affine_align(dset_from,dset_to,skull_strip=True,mask=None,suffix='_aff',prefix=None,cost=None,epi=False,resample='wsinc5',grid_size=None,opts=[]):
    ''' interface to 3dAllineate to align anatomies and EPIs '''

    dset_ss = lambda dset: os.path.split(nl.suffix(dset,'_ns'))[1]
    def dset_source(dset):
        if skull_strip==True or skull_strip==dset:
            return dset_ss(dset)
        else:
            return dset

    dset_affine = prefix
    if dset_affine==None:
        dset_affine = os.path.split(nl.suffix(dset_from,suffix))[1]
    dset_affine_mat_1D = nl.prefix(dset_affine) + '_matrix.1D'
    dset_affine_par_1D = nl.prefix(dset_affine) + '_params.1D'

    if os.path.exists(dset_affine):
        # final product already exists
        return

    for dset in [dset_from,dset_to]:
        if skull_strip==True or skull_strip==dset:
            nl.skull_strip(dset,'_ns')

    mask_use = mask
    if mask:
        # the mask was probably made in the space of the original dset_to anatomy,
        # which has now been cropped from the skull stripping. So the lesion mask
        # needs to be resampled to match the corresponding mask
        if skull_strip==True or skull_strip==dset_to:
            nl.run(['3dresample','-master',dset_u(dset_ss(dset)),'-inset',mask,'-prefix',nl.suffix(mask,'_resam')],products=nl.suffix(mask,'_resam'))
            mask_use = nl.suffix(mask,'_resam')

    all_cmd = [
        '3dAllineate',
        '-prefix', dset_affine,
        '-base', dset_source(dset_to),
        '-source', dset_source(dset_from),
        '-source_automask',
        '-1Dmatrix_save', dset_affine_mat_1D,
        '-1Dparam_save',dset_affine_par_1D,
        '-autoweight',
        '-final',resample,
        '-cmass'
    ] + opts
    if grid_size:
        all_cmd += ['-newgrid',grid_size]
    if cost:
        all_cmd += ['-cost',cost]
    if epi:
        all_cmd += ['-EPI']
    if mask:
        all_cmd += ['-emask', mask_use]

    nl.run(all_cmd,products=dset_affine)