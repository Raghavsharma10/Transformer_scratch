def to_struct(model):
    """Cast instance of model to python structure.

    :param model: Model to be casted.
    :rtype: ``dict``

    """
    model.validate()

    resp = {}
    for _, name, field in model.iterate_with_name():
        value = field.__get__(model)
        if value is None:
            continue

        value = field.to_struct(value)
        resp[name] = value
    return resp