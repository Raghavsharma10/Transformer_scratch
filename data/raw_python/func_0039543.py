def return_single_convert_numpy(self, object_id, converter, add_args=None):
        """
        Converts an object specified by the object_id into a numpy array and returns the array,
        the conversion is done by the 'converter' function

        Parameters
        ----------
        object_id : int, id of object in database
        converter : function, which takes the path of a data point and *args as parameters and returns a numpy array
        add_args : optional arguments for the converter (list/dictionary/tuple/whatever). if None, the
        converter should take only one input argument - the file path. default value: None

        Returns
        -------
        result : ndarray
        """
        return return_single_convert_numpy_base(self.dbpath, self.path_to_set, self._set_object, object_id, converter,
                                                add_args)