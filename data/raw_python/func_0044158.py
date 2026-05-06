def value(self, cell):
        """
        Extract the value of ``cell``, ready to be rendered.

        If this Column was instantiated with a ``value`` attribute, it
        is called here to provide the value. (For example, to provide a
        calculated value.) Otherwise, ``cell.value`` is returned.
        """

        if self._value is not None:
            return self._value(cell)

        else:
            return cell.value