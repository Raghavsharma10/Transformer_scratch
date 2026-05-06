def parallactic_angles(self, context):
        """ parallactic angle data source """
        # Time and antenna extents
        (lt, ut), (la, ua) = context.dim_extents('ntime', 'na')

        return (mbu.parallactic_angles(self._times[lt:ut],
                self._antenna_positions[la:ua], self._phase_dir)
                                            .reshape(context.shape)
                                            .astype(context.dtype))