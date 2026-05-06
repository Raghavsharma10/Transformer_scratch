def uuid_object(title="Reference", description="Select an object", default=None, display=True):
    """Generates a regular expression controlled UUID field"""

    uuid = {
        'pattern': '^[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{'
                   '4}-['
                   'a-fA-F0-9]{4}-[a-fA-F0-9]{12}$',
        'type': 'string',
        'title': title,
        'description': description,
    }

    if not display:
        uuid['x-schema-form'] = {
            'condition': "false"
        }

    if default is not None:
        uuid['default'] = default

    return uuid