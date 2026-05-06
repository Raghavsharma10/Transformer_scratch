def uvw(self, context):
        """ Supply UVW antenna coordinates to montblanc """

        # Shape (ntime, na, 3)
        (lt, ut), (la, ua), (l, u) = context.array_extents(context.name)

        # Create empty UVW coordinates
        data = np.empty(context.shape, context.dtype)
        data[:,:,0] = np.arange(la+1, ua+1)    # U = antenna index
        data[:,:,1] = 0                        # V = 0
        data[:,:,2] = 0                        # W = 0

        return data