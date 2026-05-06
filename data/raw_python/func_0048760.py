def save(self, fname=None, link_copy=False,raiseError=False):
        """ link_copy: only works in hfd5 format
            save space by creating link when identical arrays are found,
            it may slows down the saving (3 or 4 folds) but saves space
            when saving different dataset together (since it does not duplicate
            arrays)
        """
        if fname is None:
            fname = self.filename
        assert fname is not None
        save(fname, self, link_copy=link_copy,raiseError=raiseError)