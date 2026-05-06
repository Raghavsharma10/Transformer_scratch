def _binary_arithemtic(self, left, binary, right):
        """
        Parameters
        ----------
        operand: Column object, integer or float
            Value on which to apply operator to this column
        binary: char
            binary arithmetic operator (-, +, *, /, ^, %)

        Returns
        -------
        self

        Notes
        -----
        Returning self will allow the next object to use this column ops and
        concatenate something else
        """
        if isinstance(right, (int, float)):
            right = right
        elif isinstance(right, Column):
            right = right.execution_name
        else:
            raise AttributeError(
                "{} can only be used ".format(binary)
                + "with integer, float or column")

        if isinstance(left, (int, float)):
            left = left
        elif isinstance(left, Column):
            left = left.execution_name
        else:
            raise AttributeError(
                "{} can only be used ".format(binary)
                + "with integer, float or column")

        copy = self.copy()
        copy.query.removeSELECT("{}".format(copy.execution_name))
        if binary == '^':  # POWER needs a different treatment
            copy.execution_name = "pow({},{})".format(left, right)
        else:
            copy.execution_name = "{}{}{}".format(left, binary, right)
        copy.query.addSELECT(copy.execution_name)

        return copy