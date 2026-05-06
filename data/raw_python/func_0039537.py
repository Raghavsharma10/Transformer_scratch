def extract_feature(self, extractor, force_extraction=False, verbose=0, add_args=None, custom_name=None):
        """
        Extracts a feature and stores it in the database

        Parameters
        ----------
        extractor : function, which takes the path of a data point and *args as parameters and returns a feature
        force_extraction : boolean, if True - will re-extract feature even if a feature with this name already
        exists in the database, otherwise, will only extract if the feature doesn't exist in the database.
        default value: False
        verbose : int, if bigger than 0, will print the current number of the file for which data is being extracted
        add_args : optional arguments for the extractor (list/dictionary/tuple/whatever). if None, the
        extractor should take only one input argument - the file path. default value: None
        custom_name : string, optional name for the feature (it will be stored in the database with the custom_name
        instead of extractor function name). if None, the extractor function name will be used. default value: None

        Returns
        -------
        None
        """
        if self._prepopulated is False:
            raise errors.EmptyDatabase(self.dbpath)
        else:
            return extract_feature_base(self.dbpath, self.path_to_set, self._set_object, extractor, force_extraction,
                                        verbose, add_args, custom_name)