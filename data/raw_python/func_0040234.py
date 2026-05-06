def unpack_archive(fname,out_dir):
    '''unpacks the archive file ``fname`` and reconstructs datasets into ``out_dir``

    Datasets are reconstructed and auto-named using :meth:`create_dset`. The raw directories
    that made the datasets are archive with the dataset name suffixed by ``tgz``, and any other
    files found in the archive are put into ``other_files.tgz``'''
    with nl.notify('Unpacking archive %s' % fname):
        tmp_dir = tempfile.mkdtemp()
        tmp_unpack = os.path.join(tmp_dir,'unpack')
        os.makedirs(tmp_unpack)
        nl.utils.unarchive(fname,tmp_unpack)
        reconstruct_files(tmp_unpack)
        out_dir = os.path.abspath(out_dir)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        if not os.path.exists(tmp_unpack+'-sorted'):
            return
        with nl.run_in(tmp_unpack+'-sorted'):
            for fname in glob.glob('*.nii'):
                nl.run(['gzip',fname])
            for fname in glob.glob('*.nii.gz'):
                new_file = os.path.join(out_dir,fname)
                if not os.path.exists(new_file):
                    shutil.move(fname,new_file)
            raw_out = os.path.join(out_dir,'raw')
            if not os.path.exists(raw_out):
                os.makedirs(raw_out)
            for rawdir in os.listdir('.'):
                rawdir_tgz = os.path.join(raw_out,rawdir+'.tgz')
                if not os.path.exists(rawdir_tgz):
                    with tarfile.open(rawdir_tgz,'w:gz') as tgz:
                        tgz.add(rawdir)
        if len(os.listdir(tmp_unpack))!=0:
            # There are still raw files left
            with tarfile.open(os.path.join(raw_out,'other_files.tgz'),'w:gz') as tgz:
                tgz.add(tmp_unpack)
    shutil.rmtree(tmp_dir)