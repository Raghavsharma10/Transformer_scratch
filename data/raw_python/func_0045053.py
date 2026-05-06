def uflat(self):
        """return a flatten **copy** of the main numpy array with only the
        dependant variables.

        Be carefull, modification of these data will not be reflected on
        the main array!
        """  # noqa
        aligned_arrays = [self[key].values[[(slice(None)
                                             if c in coords
                                             else None)
                                            for c in self._coords]].T
                          for key, coords in self.dependent_variables_info]
        return np.vstack(aligned_arrays).flatten("F")