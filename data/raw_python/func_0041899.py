def align_epi_anat(anatomy,epi_dsets,skull_strip_anat=True):
    ''' aligns epis to anatomy using ``align_epi_anat.py`` script

    :epi_dsets:       can be either a string or list of strings of the epi child datasets
    :skull_strip_anat:     if ``True``, ``anatomy`` will be skull-stripped using the default method

    The default output suffix is "_al"
    '''

    if isinstance(epi_dsets,basestring):
        epi_dsets = [epi_dsets]

    if len(epi_dsets)==0:
        nl.notify('Warning: no epi alignment datasets given for anatomy %s!' % anatomy,level=nl.level.warning)
        return

    if all(os.path.exists(nl.suffix(x,'_al')) for x in epi_dsets):
        return

    anatomy_use = anatomy

    if skull_strip_anat:
        nl.skull_strip(anatomy,'_ns')
        anatomy_use = nl.suffix(anatomy,'_ns')

    inputs = [anatomy_use] + epi_dsets
    dset_products = lambda dset: [nl.suffix(dset,'_al'), nl.prefix(dset)+'_al_mat.aff12.1D', nl.prefix(dset)+'_tsh_vr_motion.1D']
    products = nl.flatten([dset_products(dset) for dset in epi_dsets])
    with nl.run_in_tmp(inputs,products):
        if nl.is_nifti(anatomy_use):
            anatomy_use = nl.afni_copy(anatomy_use)
        epi_dsets_use = []
        for dset in epi_dsets:
            if nl.is_nifti(dset):
                epi_dsets_use.append(nl.afni_copy(dset))
            else:
                epi_dsets_use.append(dset)
        cmd = ["align_epi_anat.py", "-epi2anat", "-anat_has_skull", "no", "-epi_strip", "3dAutomask","-anat", anatomy_use, "-epi_base", "5", "-epi", epi_dsets_use[0]]
        if len(epi_dsets_use)>1:
            cmd += ['-child_epi'] + epi_dsets_use[1:]
            out = nl.run(cmd)

        for dset in epi_dsets:
            if nl.is_nifti(dset):
                dset_nifti = nl.nifti_copy(nl.prefix(dset)+'_al+orig')
                if dset_nifti and os.path.exists(dset_nifti) and dset_nifti.endswith('.nii') and dset.endswith('.gz'):
                    nl.run(['gzip',dset_nifti])