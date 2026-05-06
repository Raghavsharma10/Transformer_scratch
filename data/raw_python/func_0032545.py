def _unary_arithmetic(self, unary):
        """
        Parameters
        ----------
        unary: char
            Unary arithmetic operator (-, +) to be applied to this column

        Returns
        -------
        self

        Notes
        -----
        Returning self will allow the next object to use this column ops and
        concatenate something else
        """
        copy = self.copy()
        copy.query.removeSELECT("{}".format(copy.execution_name))
        copy.execution_name = "{}({})".format(unary, self.execution_name)
        copy.query.addSELECT(copy.execution_name)

        return copy