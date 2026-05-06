def transform_generator(fn):
    """A decorator that marks transform pipes that should be called to create the real transform"""
    if six.PY2:
        fn.func_dict['is_transform_generator'] = True
    else:
        # py3
        fn.__dict__['is_transform_generator'] = True
    return fn