def _all_combos(self):
        """
        RETURN AN ITERATOR OF ALL COORDINATES
        """
        combos = _product(self.dims)
        if not combos:
            return

        calc = [(coalesce(_product(self.dims[i+1:]), 1), mm) for i, mm in enumerate(self.dims)]

        for c in xrange(combos):
            yield tuple(int(c / dd) % mm for dd, mm in calc)