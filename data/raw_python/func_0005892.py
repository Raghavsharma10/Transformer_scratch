def set_window_params(self, cols=None, rows=None):
        """Sets pty window params.

        :param int cols:
        :param int rows:

        """
        self._set_aliased('cols', cols)
        self._set_aliased('rows', rows)

        return self