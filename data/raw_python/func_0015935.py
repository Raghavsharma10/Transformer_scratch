def get_real_field(model, field_name):
    '''
    Get the real field from a model given its name.

    Handle nested models recursively (aka. ``__`` lookups)
    '''
    parts = field_name.split('__')
    field = model._meta.get_field(parts[0])
    if len(parts) == 1:
        return model._meta.get_field(field_name)
    elif isinstance(field, models.ForeignKey):
        return get_real_field(field.rel.to, '__'.join(parts[1:]))
    else:
        raise Exception('Unhandled field: %s' % field_name)