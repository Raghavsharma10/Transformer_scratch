def is_transform_generator(fn):
    """Return true of the function has been marked with @transform_generator"""
    try:
        if six.PY2:
            fn.func_dict['is_transform_generator'] = True
        else:
            # py3
            return fn.__dict__.get('is_transform_generator', False)
    except AttributeError:
        return False