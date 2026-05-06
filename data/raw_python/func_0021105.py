def convert_type(self, type):
        """Convert type to Pandas
        """

        # Mapping
        mapping = {
            'any': np.dtype('O'),
            'array': np.dtype(list),
            'boolean': np.dtype(bool),
            'date': np.dtype('O'),
            'datetime': np.dtype('datetime64[ns]'),
            'duration': np.dtype('O'),
            'geojson': np.dtype('O'),
            'geopoint': np.dtype('O'),
            'integer': np.dtype(int),
            'number': np.dtype(float),
            'object': np.dtype(dict),
            'string': np.dtype('O'),
            'time': np.dtype('O'),
            'year': np.dtype(int),
            'yearmonth': np.dtype('O'),
        }

        # Get type
        if type not in mapping:
            message = 'Type "%s" is not supported' % type
            raise tableschema.exceptions.StorageError(message)

        return mapping[type]