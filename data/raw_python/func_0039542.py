def copy_features(self, dbpath_origin, force_copy=False):
        """
        Copies features from one database to another (base object should be of the same type)

        Parameters
        ----------
        dbpath_origin : string, path to SQLite database file from which the features will be copied
        force_copy : boolean, if True - will overwrite features with same name when copying, if False, won't;
        default value: False

        Returns
        -------
        None
        """
        copy_features_base(dbpath_origin, self.dbpath, self._set_object, force_copy)
        return None