def DefaultExtension(schema_obj, form_obj, schemata=None):
    """Create a default field"""

    if schemata is None:
        schemata = ['systemconfig', 'profile', 'client']

    DefaultExtends = {
        'schema': {
            "properties/modules": [
                schema_obj
            ]
        },
        'form': {
            'modules': {
                'items/': form_obj
            }
        }
    }

    output = {}

    for schema in schemata:
        output[schema] = DefaultExtends

    return output