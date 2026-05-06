def uvw(self, context):
        """ Per-antenna UVW coordinate data source """

        # Hacky access of private member
        cube = context._cube

        # Create antenna1 source context
        a1_actual = cube.array("antenna1", reify=True)
        a1_ctx = SourceContext("antenna1", cube, context.cfg,
            context.iter_args, cube.array("antenna1"),
            a1_actual.shape, a1_actual.dtype)

        # Create antenna2 source context
        a2_actual = cube.array("antenna2", reify=True)
        a2_ctx = SourceContext("antenna2", cube, context.cfg,
            context.iter_args, cube.array("antenna2"),
            a2_actual.shape, a2_actual.dtype)

        # Get antenna1 and antenna2 data
        ant1 = self.antenna1(a1_ctx).ravel()
        ant2 = self.antenna2(a2_ctx).ravel()

        # Obtain per baseline UVW data
        lrow, urow = MS.uvw_row_extents(context)
        uvw = self._manager.ordered_uvw_table.getcol(MS.UVW,
                                                startrow=lrow,
                                                nrow=urow-lrow)

        # Perform the per-antenna UVW decomposition
        ntime, nbl = context.dim_extent_size('ntime', 'nbl')
        na = context.dim_global_size('na')
        chunks = np.repeat(nbl, ntime).astype(ant1.dtype)

        auvw = mbu.antenna_uvw(uvw, ant1, ant2, chunks, nr_of_antenna=na)

        return auvw.reshape(context.shape).astype(context.dtype)