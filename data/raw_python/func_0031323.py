def get_fields(cls):
        """
        Returns a dictionary of fields and field instances for this schema.
        """
        fields = {}
        for field_name in dir(cls):
            if isinstance(getattr(cls, field_name), Field):
                field = getattr(cls, field_name)
                field_name = field.field_name or field_name
                fields[field_name] = field
        return fields