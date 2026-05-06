def affine_apply(dset_from,affine_1D,master,affine_suffix='_aff',interp='NN',inverse=False,prefix=None):
    '''apply the 1D file from a previously aligned dataset
    Applies the matrix in ``affine_1D`` to ``dset_from`` and makes the final grid look like the dataset ``master``
    using the interpolation method ``interp``. If ``inverse`` is True, will apply the inverse of ``affine_1D`` instead'''
    affine_1D_use = affine_1D
    if inverse:
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            temp.write(subprocess.check_output(['cat_matvec',affine_1D,'-I']))
            affine_1D_use = temp.name
    if prefix==None:
        prefix = nl.suffix(dset_from,affine_suffix)
    nl.run(['3dAllineate','-1Dmatrix_apply',affine_1D_use,'-input',dset_from,'-prefix',prefix,'-master',master,'-final',interp],products=prefix)