def mask_average(dset,mask):
    '''Returns average of voxels in ``dset`` within non-zero voxels of ``mask``'''
    o = nl.run(['3dmaskave','-q','-mask',mask,dset])
    if o:
        return float(o.output.split()[-1])