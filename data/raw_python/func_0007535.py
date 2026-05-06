def instantiate_object_id_field(object_id_class_or_tuple=models.TextField):
    """
    Instantiates and returns a model field for FieldHistory.object_id.

    object_id_class_or_tuple may be either a Django model field class or a
    tuple of (model_field, kwargs), where kwargs is a dict passed to
    model_field's constructor.
    """
    if isinstance(object_id_class_or_tuple, (list, tuple)):
        object_id_class, object_id_kwargs = object_id_class_or_tuple
    else:
        object_id_class = object_id_class_or_tuple
        object_id_kwargs = {}

    if not issubclass(object_id_class, models.fields.Field):
        raise TypeError('settings.%s must be a Django model field or (field, kwargs) tuple' % OBJECT_ID_TYPE_SETTING)
    if not isinstance(object_id_kwargs, dict):
        raise TypeError('settings.%s kwargs must be a dict' % OBJECT_ID_TYPE_SETTING)

    return object_id_class(db_index=True, **object_id_kwargs)