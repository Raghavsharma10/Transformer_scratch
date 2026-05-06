def point_stokes(self, context):
        """ Supply point source stokes parameters to montblanc """

        # Shape (npsrc, ntime, 4)
        (ls, us), (lt, ut), (l, u) = context.array_extents(context.name)

        data = np.empty(context.shape, context.dtype)
        data[ls:us,:,l:u] = np.asarray(lm_stokes)[ls:us,None,:]
        return data