def qwarp_invert(warp_param_dset,output_dset,affine_1Dfile=None):
    '''inverts a qwarp (defined in ``warp_param_dset``) (and concatenates affine matrix ``affine_1Dfile`` if given)
    outputs the inverted warp + affine to ``output_dset``'''

    cmd = ['3dNwarpCat','-prefix',output_dset]

    if affine_1Dfile:
        cmd += ['-warp1','INV(%s)' % affine_1Dfile, '-warp2','INV(%s)' % warp_param_dset]
    else:
        cmd += ['-warp1','INV(%s)' % warp_param_dset]

    nl.run(cmd,products=output_dset)