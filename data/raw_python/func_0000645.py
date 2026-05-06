def restore_type(self, type):
        """Restore type from SQL
        """

        # All dialects
        mapping = {
            ARRAY: 'array',
            sa.Boolean: 'boolean',
            sa.Date: 'date',
            sa.DateTime: 'datetime',
            sa.Float: 'number',
            sa.Integer: 'integer',
            JSONB: 'object',
            JSON: 'object',
            sa.Numeric: 'number',
            sa.Text: 'string',
            sa.Time: 'time',
            sa.VARCHAR: 'string',
            UUID: 'string',
        }

        # Get field type
        field_type = None
        for key, value in mapping.items():
            if isinstance(type, key):
                field_type = value

        # Not supported
        if field_type is None:
            message = 'Type "%s" is not supported'
            raise tableschema.exceptions.StorageError(message % type)

        return field_type