def flag(self, context):
        """ Flag data source """
        lrow, urow = MS.row_extents(context)

        flag = self._manager.ordered_main_table.getcol(
            MS.FLAG, startrow=lrow, nrow=urow-lrow)

        return flag.reshape(context.shape).astype(context.dtype)