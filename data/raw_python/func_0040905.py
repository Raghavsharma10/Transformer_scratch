def nifti_copy(filename,prefix=None,gzip=True):
    ''' creates a ``.nii`` copy of the given dataset and returns the filename as a string'''
    # I know, my argument ``prefix`` clobbers the global method... but it makes my arguments look nice and clean
    if prefix==None:
        prefix = filename
    nifti_filename = globals()['prefix'](prefix) + ".nii"
    if gzip:
        nifti_filename += '.gz'
    if not os.path.exists(nifti_filename):
        try:
            subprocess.check_call(['3dAFNItoNIFTI','-prefix',nifti_filename,str(filename)])
        except subprocess.CalledProcessError:
            nl.notify('Error: could not convert "%s" to NIFTI dset!' % filename,level=nl.level.error)
            return None
    return nifti_filename