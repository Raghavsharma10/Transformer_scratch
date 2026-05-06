def point_lm(self, context):
        """ Return a lm coordinate array to montblanc """
        lm = np.empty(context.shape, context.dtype)

        # Print the array schema
        montblanc.log.info(context.array_schema.shape)
        # Print the space of iteration
        montblanc.log.info(context.iter_args)

        (ls, us) = context.dim_extents('npsrc')

        lm[:,0] = 0.0008
        lm[:,1] = 0.0036

        lm[:,:] = 0
        return lm