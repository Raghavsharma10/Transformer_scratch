def get_serializer_fields(self, path, method, view):
        """
        Return a list of `coreapi.Field` instances corresponding to any
        request body input, as determined by the serializer class.
        """
        if method not in ('PUT', 'PATCH', 'POST'):
            return []

        if not hasattr(view, 'get_serializer'):
            return []

        serializer = view.get_serializer()
        if not isinstance(serializer, Serializer):
            return []

        fields = []
        for field in serializer.fields.values():
            if field.read_only or isinstance(field, HiddenField):
                continue

            required = field.required and method != 'PATCH'
            description = force_text(field.help_text) if field.help_text else ''
            field_type = get_field_type(field)
            description += '; ' + field_type if description else field_type
            field = coreapi.Field(
                name=field.field_name,
                location='form',
                required=required,
                description=description,
                schema=schemas.field_to_schema(field),
            )
            fields.append(field)

        return fields