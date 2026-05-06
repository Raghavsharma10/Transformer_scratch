def model_fields(model, allow_pk=False, only=None, exclude=None,
                 field_args=None, converter=None):
    """
    Generate a dictionary of fields for a given Peewee model.

    See `model_form` docstring for description of parameters.
    """
    converter = converter or ModelConverter()
    field_args = field_args or {}

    model_fields = list(model._meta.sorted_fields)
    if not allow_pk:
        model_fields.pop(0)

    if only:
        model_fields = [x for x in model_fields if x.name in only]
    elif exclude:
        model_fields = [x for x in model_fields if x.name not in exclude]

    field_dict = {}
    for model_field in model_fields:
        name, field = converter.convert(
            model,
            model_field,
            field_args.get(model_field.name))
        field_dict[name] = field

    return field_dict