def restore_type(self, dtype, sample=None):
        """Restore type from Pandas
        """

        # Pandas types
        if pdc.is_bool_dtype(dtype):
            return 'boolean'
        elif pdc.is_datetime64_any_dtype(dtype):
            return 'datetime'
        elif pdc.is_integer_dtype(dtype):
            return 'integer'
        elif pdc.is_numeric_dtype(dtype):
            return 'number'

        # Python types
        if sample is not None:
            if isinstance(sample, (list, tuple)):
                return 'array'
            elif isinstance(sample, datetime.date):
                return 'date'
            elif isinstance(sample, isodate.Duration):
                return 'duration'
            elif isinstance(sample, dict):
                return 'object'
            elif isinstance(sample, six.string_types):
                return 'string'
            elif isinstance(sample, datetime.time):
                return 'time'

        return 'string'