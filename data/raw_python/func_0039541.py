def return_real_id(self):
        """
        Returns a list of real_id's

        Parameters
        ----------

        Returns
        -------
        A list of real_id values for the dataset (a real_id is the filename minus the suffix and prefix)
        """
        if self._prepopulated is False:
            raise errors.EmptyDatabase(self.dbpath)
        else:
            return return_real_id_base(self.dbpath, self._set_object)