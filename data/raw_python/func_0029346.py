def get_aggregations_fields(cls, params):
        """ Recursively get values under the 'field' key.

        Is used to get names of fields on which aggregations should be
        performed.
        """
        fields = []
        for key, val in params.items():
            if isinstance(val, dict):
                fields += cls.get_aggregations_fields(val)
            if key == 'field':
                fields.append(val)
        return fields