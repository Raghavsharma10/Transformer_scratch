def inside_brain(stat_dset,atlas=None,p=0.001):
    '''calculates the percentage of voxels above a statistical threshold inside a brain mask vs. outside it
    
    if ``atlas`` is ``None``, it will try to find ``TT_N27``'''
    atlas = find_atlas(atlas)
    if atlas==None:
        return None
    mask_dset = nl.suffix(stat_dset,'_atlasfrac')
    nl.run(['3dfractionize','-template',nl.strip_subbrick(stat_dset),'-input',nl.calc([atlas],'1+step(a-100)',datum='short'),'-preserve','-clip','0.2','-prefix',mask_dset],products=mask_dset,quiet=True,stderr=None)
    s = nl.roi_stats(mask_dset,nl.thresh(stat_dset,p))
    return 100.0 * s[2]['nzvoxels'] / (s[1]['nzvoxels'] + s[2]['nzvoxels'])