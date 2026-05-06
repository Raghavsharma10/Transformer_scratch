def observed_vis(self, context):
        """ Observed visibility data source """
        lrow, urow = MS.row_extents(context)

        data = self._manager.ordered_main_table.getcol(
            self._vis_column, startrow=lrow, nrow=urow-lrow)

        return data.reshape(context.shape).astype(context.dtype)