def generate_model_cls(config, schema, model_name, raml_resource,
                       es_based=True):
    """ Generate model class.

    Engine DB field types are determined using `type_fields` and only those
    types may be used.

    :param schema: Model schema dict parsed from RAML.
    :param model_name: String that is used as new model's name.
    :param raml_resource: Instance of ramlfications.raml.ResourceNode.
    :param es_based: Boolean indicating if generated model should be a
        subclass of Elasticsearch-based document class or not.
        It True, ESBaseDocument is used; BaseDocument is used otherwise.
        Defaults to True.
    """
    from nefertari.authentication.models import AuthModelMethodsMixin
    base_cls = engine.ESBaseDocument if es_based else engine.BaseDocument
    model_name = str(model_name)
    metaclass = type(base_cls)
    auth_model = schema.get('_auth_model', False)

    bases = []
    if config.registry.database_acls:
        from nefertari_guards import engine as guards_engine
        bases.append(guards_engine.DocumentACLMixin)
    if auth_model:
        bases.append(AuthModelMethodsMixin)
    bases.append(base_cls)

    attrs = {
        '__tablename__': model_name.lower(),
        '_public_fields': schema.get('_public_fields') or [],
        '_auth_fields': schema.get('_auth_fields') or [],
        '_hidden_fields': schema.get('_hidden_fields') or [],
        '_nested_relationships': schema.get('_nested_relationships') or [],
    }
    if '_nesting_depth' in schema:
        attrs['_nesting_depth'] = schema.get('_nesting_depth')

    # Generate fields from properties
    properties = schema.get('properties', {})
    for field_name, props in properties.items():
        if field_name in attrs:
            continue

        db_settings = props.get('_db_settings')
        if db_settings is None:
            continue

        field_kwargs = db_settings.copy()
        field_kwargs['required'] = bool(field_kwargs.get('required'))

        for default_attr_key in ('default', 'onupdate'):
            value = field_kwargs.get(default_attr_key)
            if is_callable_tag(value):
                field_kwargs[default_attr_key] = resolve_to_callable(value)

        type_name = (
            field_kwargs.pop('type', 'string') or 'string').lower()
        if type_name not in type_fields:
            raise ValueError('Unknown type: {}'.format(type_name))

        field_cls = type_fields[type_name]

        if field_cls is engine.Relationship:
            prepare_relationship(
                config, field_kwargs['document'],
                raml_resource)
        if field_cls is engine.ForeignKeyField:
            key = 'ref_column_type'
            field_kwargs[key] = type_fields[field_kwargs[key]]
        if field_cls is engine.ListField:
            key = 'item_type'
            field_kwargs[key] = type_fields[field_kwargs[key]]

        attrs[field_name] = field_cls(**field_kwargs)

    # Update model definition with methods and variables defined in registry
    attrs.update(registry.mget(model_name))

    # Generate new model class
    model_cls = metaclass(model_name, tuple(bases), attrs)
    setup_model_event_subscribers(config, model_cls, schema)
    setup_fields_processors(config, model_cls, schema)
    return model_cls, auth_model