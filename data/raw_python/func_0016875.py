def antenna2(self, context):
        """ antenna2 data source """
        lrow, urow = MS.uvw_row_extents(context)
        antenna2 = self._manager.ordered_uvw_table.getcol(
            MS.ANTENNA2, startrow=lrow, nrow=urow-lrow)

        return antenna2.reshape(context.shape).astype(context.dtype)