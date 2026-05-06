def link(self, var):
        """ Link this Varying to another object from which it will derive its
        dtype. This method is used internally when assigning an attribute to
        a varying using syntax ``Function[varying] = attr``.
        """
        assert self._dtype is not None or hasattr(var, 'dtype')
        self._link = var
        self.changed()