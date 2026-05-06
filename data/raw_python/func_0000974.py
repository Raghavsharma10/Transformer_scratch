def get_form_schema(form):
    """Return a JSON Schema object for a Django Form."""
    schema = {
        'type': 'object',
        'properties': {},
    }

    for name, field in form.base_fields.items():
        schema['properties'][name] = get_field_schema(name, field)
        if field.required:
            schema.setdefault('required', []).append(name)

    return schema