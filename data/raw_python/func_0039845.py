def mgz_to_nifti(filename,prefix=None,gzip=True):
    '''Convert ``filename`` to a NIFTI file using ``mri_convert``'''
    setup_freesurfer()
    if prefix==None:
        prefix = nl.prefix(filename) + '.nii'
    if gzip and not prefix.endswith('.gz'):
        prefix += '.gz'
    nl.run([os.path.join(freesurfer_home,'bin','mri_convert'),filename,prefix],products=prefix)