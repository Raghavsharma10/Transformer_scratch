def get_filename(self, variable):
        """Return the auxiliary file name the given variable is allocated
        to or |None| if the given variable is not allocated to any
        auxiliary file name.

        >>> from hydpy import dummies
        >>> eqb = dummies.v2af.eqb[0]
        >>> dummies.v2af.get_filename(eqb)
        'file1'
        >>> eqb += 500.0
        >>> dummies.v2af.get_filename(eqb)
        """
        fn2var = self._type2filename2variable.get(type(variable), {})
        for (fn_, var) in fn2var.items():
            if var == variable:
                return fn_
        return None