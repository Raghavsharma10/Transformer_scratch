def add_field(self, field_instance):
        """
        Appends a field.
        """
        if isinstance(field_instance, BaseScriptField):
            field_instance = field_instance
        else:
            raise ValueError('Expected a basetring or Field instance')

        self.fields.append(field_instance)

        return self