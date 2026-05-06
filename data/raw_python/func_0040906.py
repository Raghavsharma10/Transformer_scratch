def dset_copy(dset,to_dir):
    '''robust way to copy a dataset (including AFNI briks)'''
    if nl.is_afni(dset):
        dset_strip = re.sub(r'\.(HEAD|BRIK)?(\.(gz|bz))?','',dset)
        for dset_file in [dset_strip + '.HEAD'] + glob.glob(dset_strip + '.BRIK*'):
            if os.path.exists(dset_file):
                shutil.copy(dset_file,to_dir)
    else:
        if os.path.exists(dset):
            shutil.copy(dset,to_dir)
        else:
            nl.notify('Warning: couldn\'t find file %s to copy to %s' %(dset,to_dir),level=nl.level.warning)