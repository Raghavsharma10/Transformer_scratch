def _get_fields_for_model(model):
    """
    Gets all of the fields on the model.

    :param DeclarativeModel model: A SQLAlchemy ORM Model
    :return: A tuple of the fields on the Model corresponding
        to the columns on the Model.
    :rtype: tuple
    """
    fields = []
    for name in model._sa_class_manager:
        prop = getattr(model, name)
        if isinstance(prop.property, RelationshipProperty):
            for pk in prop.property.mapper.primary_key:
                fields.append('{0}.{1}'.format(name, pk.name))
        else:
            fields.append(name)
    return tuple(fields)