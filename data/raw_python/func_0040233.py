def reconstruct_files(input_dir):
    '''sorts ``input_dir`` and tries to reconstruct the subdirectories found'''
    input_dir = input_dir.rstrip('/')
    with nl.notify('Attempting to organize/reconstruct directory'):
        # Some datasets start with a ".", which confuses many programs
        for r,ds,fs in os.walk(input_dir):
            for f in fs:
                if f[0]=='.':
                    shutil.move(os.path.join(r,f),os.path.join(r,'i'+f))
        nl.dicom.organize_dir(input_dir)
        output_dir = '%s-sorted' % input_dir
        if os.path.exists(output_dir):
            with nl.run_in(output_dir):
                for dset_dir in os.listdir('.'):
                    with nl.notify('creating dataset from %s' % dset_dir):
                        nl.dicom.create_dset(dset_dir)
        else:
            nl.notify('Warning: failed to auto-organize directory %s' % input_dir,level=nl.level.warning)