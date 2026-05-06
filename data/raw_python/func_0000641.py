def convert_type(self, type):
        """Convert type to SQL
        """

        # Default dialect
        mapping = {
            'any': sa.Text,
            'array': None,
            'boolean': sa.Boolean,
            'date': sa.Date,
            'datetime': sa.DateTime,
            'duration': None,
            'geojson': None,
            'geopoint': None,
            'integer': sa.Integer,
            'number': sa.Float,
            'object': None,
            'string': sa.Text,
            'time': sa.Time,
            'year': sa.Integer,
            'yearmonth': None,
        }

        # Postgresql dialect
        if self.__dialect == 'postgresql':
            mapping.update({
                'array': JSONB,
                'geojson': JSONB,
                'number': sa.Numeric,
                'object': JSONB,
            })

        # Not supported type
        if type not in mapping:
            message = 'Field type "%s" is not supported'
            raise tableschema.exceptions.StorageError(message % type)

        return mapping[type]