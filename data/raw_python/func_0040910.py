def dset_grids_equal(dsets):
    '''Tests if each dataset in the ``list`` ``dsets`` has the same number of voxels and voxel-widths'''
    infos = [dset_info(dset) for dset in dsets]
    for i in xrange(3):
        if len(set([x.voxel_size[i] for x in infos]))>1 or len(set([x.voxel_dims[i] for x in infos]))>1:
            return False
    return True