def return_multiple_convert_numpy(self, start_id, end_id, converter, add_args=None):
        """
        Converts several objects, with ids in the range (start_id, end_id)
        into a 2d numpy array and returns the array, the conversion is done by the 'converter' function

        Parameters
        ----------
        start_id : the id of the first object to be converted
        end_id : the id of the last object to be converted, if equal to -1, will convert all data points in range
        (start_id, <id of last element in database>)
        converter : function, which takes the path of a data point and *args as parameters and returns a numpy array
        add_args : optional arguments for the converter (list/dictionary/tuple/whatever). if None, the
        converter should take only one input argument - the file path. default value: None

        Returns
        -------
        result : 2-dimensional ndarray
        """
        if end_id == -1:
            end_id = self.points_amt
        return return_multiple_convert_numpy_base(self.dbpath, self.path_to_set, self._set_object, start_id, end_id,
                                                  converter, add_args)