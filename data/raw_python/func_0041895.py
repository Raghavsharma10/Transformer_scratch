def qwarp_align(dset_from,dset_to,skull_strip=True,mask=None,affine_suffix='_aff',suffix='_qwarp',prefix=None):
    '''aligns ``dset_from`` to ``dset_to`` using 3dQwarp

    Will run ``3dSkullStrip`` (unless ``skull_strip`` is ``False``), ``3dUnifize``,
    ``3dAllineate``, and then ``3dQwarp``. This method will add suffixes to the input
    dataset for the intermediate files (e.g., ``_ss``, ``_u``). If those files already
    exist, it will assume they were intelligently named, and use them as is

    :skull_strip:       If True/False, turns skull-stripping of both datasets on/off.
                        If a string matching ``dset_from`` or ``dset_to``, will only
                        skull-strip the given dataset
    :mask:              Applies the given mask to the alignment. Because of the nature
                        of the alignment algorithms, the mask is **always** applied to
                        the ``dset_to``. If this isn't what you want, you need to reverse
                        the transform and re-apply it (e.g., using :meth:`qwarp_invert`
                        and :meth:`qwarp_apply`). If the ``dset_to`` dataset is skull-stripped,
                        the mask will also be resampled to match the ``dset_to`` grid.
    :affine_suffix:     Suffix applied to ``dset_from`` to name the new dataset, as well as
                        the ``.1D`` file.
    :suffix:            Suffix applied to the final ``dset_from`` dataset. An additional file
                        with the additional suffix ``_WARP`` will be created containing the parameters
                        (e.g., with the default ``_qwarp`` suffix, the parameters will be in a file with
                        the suffix ``_qwarp_WARP``)
    :prefix:            Alternatively to ``suffix``, explicitly give the full output filename

    The output affine dataset and 1D, as well as the output of qwarp are named by adding
    the given suffixes (``affine_suffix`` and ``qwarp_suffix``) to the ``dset_from`` file

    If ``skull_strip`` is a string instead of ``True``/``False``, it will only skull strip the given
    dataset instead of both of them

    # TODO: currently does not work with +tlrc datasets because the filenames get mangled
    '''

    dset_ss = lambda dset: os.path.split(nl.suffix(dset,'_ns'))[1]
    dset_u = lambda dset: os.path.split(nl.suffix(dset,'_u'))[1]
    def dset_source(dset):
        if skull_strip==True or skull_strip==dset:
            return dset_ss(dset)
        else:
            return dset

    dset_affine = os.path.split(nl.suffix(dset_from,affine_suffix))[1]
    dset_affine_1D = nl.prefix(dset_affine) + '.1D'
    dset_qwarp = prefix
    if dset_qwarp==None:
        dset_qwarp = os.path.split(nl.suffix(dset_from,suffix))[1]

    if os.path.exists(dset_qwarp):
        # final product already exists
        return

    affine_align(dset_from,dset_to,skull_strip,mask,affine_suffix)

    for dset in [dset_from,dset_to]:
        nl.run([
            '3dUnifize',
            '-prefix', dset_u(dset_source(dset)),
            '-input', dset_source(dset)
        ],products=[dset_u(dset_source(dset))])

    mask_use = mask
    if mask:
        # the mask was probably made in the space of the original dset_to anatomy,
        # which has now been cropped from the skull stripping. So the lesion mask
        # needs to be resampled to match the corresponding mask
        if skull_strip==True or skull_strip==dset_to:
            nl.run(['3dresample','-master',dset_u(dset_ss(dset)),'-inset',mask,'-prefix',nl.suffix(mask,'_resam')],products=nl.suffix(mask,'_resam'))
            mask_use = nl.suffix(mask,'_resam')

    warp_cmd = [
        '3dQwarp',
        '-prefix', dset_qwarp,
        '-duplo', '-useweight', '-blur', '0', '3',
        '-iwarp',
        '-base', dset_u(dset_source(dset_to)),
        '-source', dset_affine
    ]

    if mask:
        warp_cmd += ['-emask', mask_use]

    nl.run(warp_cmd,products=dset_qwarp)