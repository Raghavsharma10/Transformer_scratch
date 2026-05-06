def organize_dir(orig_dir):
    '''scans through the given directory and organizes DICOMs that look similar into subdirectories

    output directory is the ``orig_dir`` with ``-sorted`` appended to the end'''

    tags = [
        (0x10,0x20),    # Subj ID
        (0x8,0x21),     # Date
        (0x8,0x31),     # Time
        (0x8,0x103e)    # Descr
    ]
    orig_dir = orig_dir.rstrip('/')
    files = scan_dir(orig_dir,tags=tags,md5_hash=True)
    dups = find_dups(files)
    for dup in dups:
        nl.notify('Found duplicates of %s...' % dup[0])
        for each_dup in dup[1:]:
            nl.notify('\tdeleting %s' % each_dup)
            try:
                os.remove(each_dup)
            except IOError:
                nl.notify('\t[failed]')
            del(files[each_dup])

    clustered = cluster_files(files)
    output_dir = '%s-sorted' % orig_dir
    for key in clustered:
        if (0x8,0x31) in clustered[key]['info']:
            clustered[key]['info'][(0x8,0x31)] = str(int(float(clustered[key]['info'][(0x8,0x31)])))
        for t in tags:
            if t not in clustered[key]['info']:
                clustered[key]['info'][t] = '_'
        run_name = '-'.join([scrub_fname(str(clustered[key]['info'][x])) for x in tags])+'-%d_images' %len(clustered[key]['files'])
        run_dir = os.path.join(output_dir,run_name)
        nl.notify('Moving files into %s' % run_dir)
        try:
            if not os.path.exists(run_dir):
                os.makedirs(run_dir)
        except IOError:
            nl.notify('Error: failed to create directory %s' % run_dir)
        else:
            for f in clustered[key]['files']:
                try:
                    dset_fname = os.path.split(f)[1]
                    if dset_fname[0]=='.':
                        dset_fname = '_' + dset_fname[1:]
                    os.rename(f,os.path.join(run_dir,dset_fname))
                except (IOError, OSError):
                    pass
    for r,ds,fs in os.walk(output_dir,topdown=False):
        for d in ds:
            dname = os.path.join(r,d)
            if len(os.listdir(dname))==0:
                os.remove(dname)