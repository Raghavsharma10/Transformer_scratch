def setup_fields_processors(config, model_cls, schema):
    """ Set up model fields' processors.

    :param config: Pyramid Configurator instance.
    :param model_cls: Model class for field of which processors should be
        set up.
    :param schema: Dict of model JSON schema.
    """
    properties = schema.get('properties', {})
    for field_name, props in properties.items():
        if not props:
            continue

        processors = props.get('_processors')
        backref_processors = props.get('_backref_processors')

        if processors:
            processors = [resolve_to_callable(val) for val in processors]
            setup_kwargs = {'model': model_cls, 'field': field_name}
            config.add_field_processors(processors, **setup_kwargs)

        if backref_processors:
            db_settings = props.get('_db_settings', {})
            is_relationship = db_settings.get('type') == 'relationship'
            document = db_settings.get('document')
            backref_name = db_settings.get('backref_name')
            if not (is_relationship and document and backref_name):
                continue

            backref_processors = [
                resolve_to_callable(val) for val in backref_processors]
            setup_kwargs = {
                'model': engine.get_document_cls(document),
                'field': backref_name
            }
            config.add_field_processors(
                backref_processors, **setup_kwargs)