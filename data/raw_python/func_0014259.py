def convert_id_to_model(value, parameter):
    '''
    Converts to a Model object.
        '', '-', '0', None convert to parameter default
        Anything else is assumed an object id and sent to `.get(id=value)`.
    '''
    value = _check_default(value, parameter, ( '', '-', '0', None ))
    if isinstance(value, (int, str)):  # only convert if we have the id
        try:
            return parameter.type.objects.get(id=value)
        except (MultipleObjectsReturned, ObjectDoesNotExist) as e:
            raise ValueError(str(e))
    return value