def add_field(self, field_instance_or_string):
        """
        Appends a field, can be a :class:`~es_fluent.fields.Field` or string.
        """
        if isinstance(field_instance_or_string, basestring):
            field_instance = Field(field_instance_or_string)
        elif isinstance(field_instance_or_string, Field):
            field_instance_or_string = field_instance
        else:
            raise ValueError('Expected a basetring or Field instance')

        self.fields.append(field_instance)

        return self