def ijk_to_xyz(dset,ijk):
    '''convert the dset indices ``ijk`` to RAI coordinates ``xyz``'''
    i = nl.dset_info(dset)
    orient_codes = [int(x) for x in nl.run(['@AfniOrient2RAImap',i.orient]).output.split()]
    orient_is = [abs(x)-1 for x in orient_codes]
    rai = []
    for rai_i in xrange(3):
         ijk_i = orient_is[rai_i]
         if orient_codes[rai_i] > 0:
             rai.append(ijk[ijk_i]*i.voxel_size[rai_i] + i.spatial_from[rai_i])
         else:
             rai.append(i.spatial_to[rai_i] - ijk[ijk_i]*i.voxel_size[rai_i])
    return rai