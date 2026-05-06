def qwarp_apply(dset_from,dset_warp,affine=None,warp_suffix='_warp',master='WARP',interp=None,prefix=None):
    '''applies the transform from a previous qwarp

    Uses the warp parameters from the dataset listed in
    ``dset_warp`` (usually the dataset name ends in ``_WARP``)
    to the dataset ``dset_from``. If a ``.1D`` file is given
    in the ``affine`` parameter, it will be applied simultaneously
    with the qwarp.

    If the parameter ``interp`` is given, will use as interpolation method,
    otherwise it will just use the default (currently wsinc5)

    Output dataset with have the ``warp_suffix`` suffix added to its name
    '''
    out_dset = prefix
    if out_dset==None:
        out_dset = os.path.split(nl.suffix(dset_from,warp_suffix))[1]
    dset_from_info = nl.dset_info(dset_from)
    dset_warp_info = nl.dset_info(dset_warp)
    if(dset_from_info.orient!=dset_warp_info.orient):
        # If the datasets are different orientations, the transform won't be applied correctly
        nl.run(['3dresample','-orient',dset_warp_info.orient,'-prefix',nl.suffix(dset_from,'_reorient'),'-inset',dset_from],products=nl.suffix(dset_from,'_reorient'))
        dset_from = nl.suffix(dset_from,'_reorient')
    warp_opt = str(dset_warp)
    if affine:
        warp_opt += ' ' + affine
    cmd = [
        '3dNwarpApply',
        '-nwarp', warp_opt]
    cmd += [
        '-source', dset_from,
        '-master',master,
        '-prefix', out_dset
    ]

    if interp:
        cmd += ['-interp',interp]

    nl.run(cmd,products=out_dset)