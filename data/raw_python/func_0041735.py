def atlas_overlap(dset,atlas=None):
    '''aligns ``dset`` to the TT_N27 atlas and returns ``(cost,overlap)``'''
    atlas = find_atlas(atlas)
    if atlas==None:
        return None
    
    cost_func = 'crM'
    infile = os.path.abspath(dset)
    tmpdir = tempfile.mkdtemp()
    with nl.run_in(tmpdir):
        o = nl.run(['3dAllineate','-verb','-base',atlas,'-source',infile + '[0]','-NN','-final','NN','-cost',cost_func,'-nmatch','20%','-onepass','-fineblur','2','-cmass','-prefix','test.nii.gz'])
        m = re.search(r'Final\s+cost = ([\d.]+) ;',o.output)
        if m:
            cost = float(m.group(1))
        o = nl.run(['3dmaskave','-mask',atlas,'-q','test.nii.gz'],stderr=None)
        data_thresh = float(o.output) / 4
        i = nl.dset_info('test.nii.gz')
        o = nl.run(['3dmaskave','-q','-mask','SELF','-sum',nl.calc([atlas,'test.nii.gz'],'equals(step(a-10),step(b-%.2f))'%data_thresh)],stderr=None)
        overlap = 100*float(o.output) / (i.voxel_dims[0]*i.voxel_dims[1]*i.voxel_dims[2])
    try:
        shutil.rmtree(tmpdir)
    except:
        pass
    return (cost,overlap)