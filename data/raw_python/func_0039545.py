def dump_feature(self, feature_name, feature, force_extraction=True):
        """
        Dumps a list of lists or ndarray of features into database (allows to
        copy features from a pre-existing .txt/.csv/.whatever file, for example)

        Parameters
        ----------
        feature : list of lists or ndarray, contains the data to be written to the database
        force_extraction : boolean, if True - will overwrite any existing feature with this name
        default value: False

        Returns
        -------
        None
        """
        dump_feature_base(self.dbpath, self._set_object, self.points_amt, feature_name, feature, force_extraction)
        return None