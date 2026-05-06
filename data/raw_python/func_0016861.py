def model_vis(self, context):
        """ model visibility data sink """
        column = self._vis_column
        msshape = None

        # Do we have a column descriptor for the supplied column?
        try:
            coldesc = self._manager.column_descriptors[column]
        except KeyError as e:
            coldesc = None

        # Try to get the shape from the descriptor
        if coldesc is not None:
            try:
                msshape = [-1] + coldesc['shape'].tolist()
            except KeyError as e:
                msshape = None

        # Otherwise guess it and warn
        if msshape is None:
            guessed_shape = [self._manager._nchan, 4]

            montblanc.log.warn("Could not obtain 'shape' from the '{c}' "
                "column descriptor. Guessing it is '{gs}'.".format(
                    c=column, gs=guessed_shape))

            msshape = [-1] + guessed_shape

        lrow, urow = MS.row_extents(context)

        self._manager.ordered_main_table.putcol(column,
            context.data.reshape(msshape),
            startrow=lrow, nrow=urow-lrow)