def register_model_converter(model, name=None, field='pk', base=IntConverter, queryset=None):
    """
    Registers a custom path converter for a model.

    :param model: a Django model
    :param str name: name to register the converter as
    :param str field: name of the lookup field
    :param base: base path converter, either by name or as class
                 (optional, defaults to `django.urls.converter.IntConverter`)
    :param queryset: a custom querset to use (optional, defaults to `model.objects.all()`)
    """
    if name is None:
        name = camel_to_snake(model.__name__)
        converter_name = '{}Converter'.format(model.__name__)
    else:
        converter_name = '{}Converter'.format(snake_to_camel(name))

    if isinstance(base, str):
        base = get_converter(base).__class__

    converter_class = type(
        converter_name,
        (ModelConverterMixin, base,),
        {'model': model, 'field': field, 'queryset': queryset}
    )

    register_converter(converter_class, name)