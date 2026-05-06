def default_antenna1(self, context):
    """ Default antenna1 values """
    ant1, ant2 = default_base_ant_pairs(self, context)
    (tl, tu), (bl, bu) = context.dim_extents('ntime', 'nbl')
    ant1_result = np.empty(context.shape, context.dtype)
    ant1_result[:,:] = ant1[np.newaxis,bl:bu]
    return ant1_result