def field_definition_post_save(sender, instance, created, raw, **kwargs):
    """
    This signal is connected by all FieldDefinition subclasses
    see comment in FieldDefinitionBase for more details
    """
    model_class = instance.model_def.model_class().render_state()
    field = instance.construct_for_migrate()
    field.model = model_class
    if created:
        if hasattr(instance._state, '_creation_default_value'):
            field.default = instance._state._creation_default_value
            delattr(instance._state, '_creation_default_value')
        add_column = popattr(instance._state, '_add_column', True)
        if add_column:
            perform_ddl('add_field', model_class, field)
            # If the field definition is raw we must re-create the model class
            # since ModelDefinitionAttribute.save won't be called
            if raw:
                instance.model_def.model_class().mark_as_obsolete()
    else:
        old_field = instance._state._pre_save_field
        delattr(instance._state, '_pre_save_field')
        perform_ddl('alter_field', model_class, old_field, field, strict=True)