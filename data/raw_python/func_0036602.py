def _use_methods(cls):
        """
        Add use_* methods for calculations.

        Code below enables the usage
        my_calculation.use_parameters(my_parameters)
        """
        use_dict = JobCalculation._use_methods
        use_dict.update({
            "parameters": {
                'valid_types': RipsDistanceMatrixParameters,
                'additional_parameter': None,
                'linkname': 'parameters',
                'docstring': 'add command line parameters',
            },
            "distance_matrix": {
                'valid_types': SinglefileData,
                'additional_parameter': None,
                'linkname': 'distance_matrix',
                'docstring': "distance matrix of point cloud",
            },
            "remote_folder": {
                'valid_types': RemoteData,
                'additional_parameter': None,
                'linkname': 'remote_folder',
                'docstring': "remote folder containing distance matrix",
            },
        })
        return use_dict