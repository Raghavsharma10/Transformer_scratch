def concat(self, other, strict=False):
        """Concats two metadata objects together.

        Parameters
        ----------
        other : Meta
            Meta object to be concatenated
        strict : bool
            if True, ensure there are no duplicate variable names

        Notes
        -----
        Uses units and name label of self if other is different
        
        Returns
        -------
        Meta
            Concatenated object
        """

        mdata = self.copy()
        # checks
        if strict:
            for key in other.keys():
                if key in mdata:
                    raise RuntimeError('Duplicated keys (variable names) ' +
                                       'across Meta objects in keys().')
            for key in other.keys_nD():
                if key in mdata:

                    raise RuntimeError('Duplicated keys (variable names) across '
                                        'Meta objects in keys_nD().')
                                        
        # make sure labels between the two objects are the same
        other_updated = self.apply_default_labels(other)
        # concat 1D metadata in data frames to copy of
        # current metadata
# <<<<<<< ho_meta_fix
        for key in other_updated.keys():
            mdata.data.loc[key] = other.data.loc[key]
        # add together higher order data
        for key in other_updated.keys_nD():
            mdata.ho_data[key] = other.ho_data[key]
# =======
#         for key in other_updated.keys():
#             mdata[key] = other_updated[key]
#         # add together higher order data
#         for key in other_updated.keys_nD():
#             mdata[key] = other_updated[key]

        return mdata