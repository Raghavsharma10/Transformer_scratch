def qwarp_epi(dset,align_subbrick=5,suffix='_qwal',prefix=None):
    '''aligns an EPI time-series using 3dQwarp

    Very expensive and not efficient at all, but it can produce pretty impressive alignment for EPI time-series with significant
    distortions due to motion'''
    info = nl.dset_info(dset)
    if info==None:
        nl.notify('Error reading dataset "%s"' % (dset),level=nl.level.error)
        return False
    if prefix==None:
        prefix = nl.suffix(dset,suffix)
    dset_sub = lambda x: '_tmp_qwarp_epi-%s_%d.nii.gz' % (nl.prefix(dset),x)
    try:
        align_dset = nl.suffix(dset_sub(align_subbrick),'_warp')
        nl.calc('%s[%d]' % (dset,align_subbrick),expr='a',prefix=align_dset,datum='float')
        for i in xrange(info.reps):
            if i != align_subbrick:
                nl.calc('%s[%d]' % (dset,i),expr='a',prefix=dset_sub(i),datum='float')
                nl.run([
                    '3dQwarp', '-nowarp',
                    '-workhard', '-superhard', '-minpatch', '9', '-blur', '0',
                    '-pear', '-nopenalty',
                    '-base', align_dset,
                    '-source', dset_sub(i),
                    '-prefix', nl.suffix(dset_sub(i),'_warp')
                ],quiet=True)
        cmd = ['3dTcat','-prefix',prefix]
        if info.TR:
            cmd += ['-tr',info.TR]
        if info.slice_timing:
            cmd += ['-tpattern',info.slice_timing]
        cmd += [nl.suffix(dset_sub(i),'_warp') for i in xrange(info.reps)]
        nl.run(cmd,quiet=True)
    except Exception as e:
        raise e
    finally:
        for i in xrange(info.reps):
            for suffix in ['','warp']:
                try:
                    os.remove(nl.suffix(dset_sub(i),suffix))
                except:
                    pass