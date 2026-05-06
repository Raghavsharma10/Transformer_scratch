def _get_field_comment(field, separator=' - '):
    """
    Create SQL comment from field's title and description

    :param field: tableschema-py Field, with optional 'title' and 'description' values
    :param separator:
    :return:

    >>> _get_field_comment(tableschema.Field({'title': 'my_title', 'description': 'my_desc'}))
    'my_title - my_desc'
    >>> _get_field_comment(tableschema.Field({'title': 'my_title', 'description': None}))
    'my_title'
    >>> _get_field_comment(tableschema.Field({'title': '', 'description': 'my_description'}))
    'my_description'
    >>> _get_field_comment(tableschema.Field({}))
    ''
    """
    title = field.descriptor.get('title') or ''
    description = field.descriptor.get('description') or ''
    return _get_comment(description, title, separator)