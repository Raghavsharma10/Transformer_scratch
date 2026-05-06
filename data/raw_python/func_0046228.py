def resource_definition(resource):
    """
    Generate a `Swagger Definitions Object <http://swagger.io/specification/#definitionsObject>`_
    from a resource.

    """
    meta = getmeta(resource)

    definition = {
        'type': "object",
        'properties': {}
    }

    for field in meta.all_fields:
        field_definition = {}

        type_def = map_field_to_type(field)
        if type_def:
            field_definition['type'] = str(type_def)
            if type_def.format:
                field_definition['format'] = type_def.format

        if field.doc_text:
            field_definition['description'] = field.doc_text

        if isinstance(field, VirtualField) or field in meta.readonly_fields:
            field_definition['readOnly'] = True

        # Use getattr to support calculated fields
        if getattr(field, 'choices', None):
            field_definition['enum'] = [c[0] for c in field.choices]

        definition['properties'][field.name] = field_definition

    return definition