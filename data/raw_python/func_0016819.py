def point_lm(self, context):
        """ Supply point source lm coordinates to montblanc """

        # Shape (npsrc, 2)
        (ls, us), _ = context.array_extents(context.name)
        return np.asarray(lm_coords[ls:us], dtype=context.dtype)