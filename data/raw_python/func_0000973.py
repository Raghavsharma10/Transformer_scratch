def get_field_schema(name, field):
    """Returns a JSON Schema representation of a form field."""
    field_schema = {
        'type': 'string',
    }

    if field.label:
        field_schema['title'] = str(field.label)  # force translation

    if field.help_text:
        field_schema['description'] = str(field.help_text)  # force translation

    if isinstance(field, (fields.URLField, fields.FileField)):
        field_schema['format'] = 'uri'
    elif isinstance(field, fields.EmailField):
        field_schema['format'] = 'email'
    elif isinstance(field, fields.DateTimeField):
        field_schema['format'] = 'date-time'
    elif isinstance(field, fields.DateField):
        field_schema['format'] = 'date'
    elif isinstance(field, (fields.DecimalField, fields.FloatField)):
        field_schema['type'] = 'number'
    elif isinstance(field, fields.IntegerField):
        field_schema['type'] = 'integer'
    elif isinstance(field, fields.NullBooleanField):
        field_schema['type'] = 'boolean'
    elif isinstance(field.widget, widgets.CheckboxInput):
        field_schema['type'] = 'boolean'

    if getattr(field, 'choices', []):
        field_schema['enum'] = sorted([choice[0] for choice in field.choices])

    # check for multiple values
    if isinstance(field.widget, (widgets.Select, widgets.ChoiceWidget)):
        if field.widget.allow_multiple_selected:
            # promote to array of <type>, move details into the items field
            field_schema['items'] = {
                'type': field_schema['type'],
            }
            if 'enum' in field_schema:
                field_schema['items']['enum'] = field_schema.pop('enum')
            field_schema['type'] = 'array'

    return field_schema