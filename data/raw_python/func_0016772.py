def unpack_from_dynamodb(*, attrs, expected, model=None, obj=None, engine=None, context=None, **kwargs):
    """Push values by dynamo_name into an object"""
    context = context or {"engine": engine}
    engine = engine or context.get("engine", None)
    if not engine:
        raise ValueError("You must provide engine or a context with an engine.")
    if model is None and obj is None:
        raise ValueError("You must provide a model or obj to unpack.")
    if model is not None and obj is not None:
        raise ValueError("Only specify model or obj.")
    if model:
        obj = model.Meta.init()

    for column in expected:
        value = attrs.get(column.dynamo_name, None)
        value = engine._load(column.typedef, value, context=context, **kwargs)
        setattr(obj, column.name, value)
    return obj