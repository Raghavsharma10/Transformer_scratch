def get_entity_description(entity):
    """
    Returns description in format:
    * entity human readable name
     * docstring
    """

    try:
        entity_name = entity.__name__.strip('_')
    except AttributeError:
        # entity is a class instance
        entity_name = entity.__class__.__name__

    label = '* %s' % formatting.camelcase_to_spaces(entity_name)
    if entity.__doc__ is not None:
        entity_docstring = formatting.dedent(smart_text(entity.__doc__)).replace('\n', '\n\t')
        return '%s\n * %s' % (label, entity_docstring)

    return label