def _comparison(self, value, operator):
        """
        Parameters
        ----------
        value: Column object or base type
            The value against which to compare the column. It can either be
            another column or a base type value (e.g. int)

        Returns
        -------
        self.query

        Notes
        -----
        Returning self.query will allow the next object to use this column
        ops and concatenate something else
        """
        if isinstance(value, Column):
            self.query.addWHERE("(({}){}({}))".format(
                self.execution_name,
                operator,
                value.execution_name))
        elif isinstance(value, str):
            self.query.addWHERE("(({}){}\'{}\')".format(
                self.execution_name,
                operator,
                value))
        else:
            self.query.addWHERE("(({}){}({}))".format(
                self.execution_name,
                operator,
                value))

        copy = self.copy()
        copy.query.removeSELECT("{}".format(copy.execution_name))
        return copy.query