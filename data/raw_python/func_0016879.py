def weight(self, context):
        """ Weight data source """
        lrow, urow = MS.row_extents(context)

        weight = self._manager.ordered_main_table.getcol(
            MS.WEIGHT, startrow=lrow, nrow=urow-lrow)

        # WEIGHT is applied across all channels
        weight = np.repeat(weight, self._manager.channels_per_band, 0)
        return weight.reshape(context.shape).astype(context.dtype)