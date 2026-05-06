def align_epi(anatomy,epis,suffix='_al',base=3,skull_strip=True):
    '''[[currently in progress]]: a simple replacement for the ``align_epi_anat.py`` script, because I've found it to be unreliable, in my usage'''
    for epi in epis:
        nl.tshift(epi,suffix='_tshift')
        nl.affine_align(nl.suffix(epi,'_tshift'),'%s[%d]'%(epis[0],base),skull_strip=False,epi=True,cost='crM',resample='wsinc5',grid_size=nl.dset_info(epi).voxel_size[0],suffix='_al')
    ss = [anatomy] if skull_strip else False
    nl.affine_align(anatomy,'%s[%d]'%(epis[0],base),skull_strip=ss,cost='lpa',grid_size=1,opts=['-interp','cubic'],suffix='_al-to-EPI')