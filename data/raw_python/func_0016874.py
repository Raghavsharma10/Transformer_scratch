def antenna1(self, context):
        """ antenna1 data source """
        lrow, urow = MS.uvw_row_extents(context)
        antenna1 = self._manager.ordered_uvw_table.getcol(
            MS.ANTENNA1, startrow=lrow, nrow=urow-lrow)

        return antenna1.reshape(context.shape).astype(context.dtype)