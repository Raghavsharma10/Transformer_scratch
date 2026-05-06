def default_antenna2(self, context):
    """ Default antenna2 values """
    ant1, ant2 = default_base_ant_pairs(self, context)
    (tl, tu), (bl, bu) = context.dim_extents('ntime', 'nbl')
    ant2_result = np.empty(context.shape, context.dtype)
    ant2_result[:,:] = ant2[np.newaxis,bl:bu]
    return ant2_result