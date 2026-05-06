def get_validation_description(view, method):
    """
    Returns validation description in format:
    ### Validation:
    validate method docstring
    * field1 name
     * field1 validation docstring
    * field2 name
     * field2 validation docstring
    """
    if method not in ('PUT', 'PATCH', 'POST') or not hasattr(view, 'get_serializer'):
        return ''

    serializer = view.get_serializer()
    description = ''
    if hasattr(serializer, 'validate') and serializer.validate.__doc__ is not None:
        description += formatting.dedent(smart_text(serializer.validate.__doc__))

    for field in serializer.fields.values():
        if not hasattr(serializer, 'validate_' + field.field_name):
            continue

        field_validation = getattr(serializer, 'validate_' + field.field_name)

        if field_validation.__doc__ is not None:
            docstring = formatting.dedent(smart_text(field_validation.__doc__)).replace('\n', '\n\t')
            field_description = '* %s\n * %s' % (field.field_name, docstring)
            description += '\n' + field_description if description else field_description

    return '### Validation:\n' + description if description else ''