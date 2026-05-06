def dumps(obj, *args, **kwargs):
    ''' Typeless dump an object to json string '''
    return json.dumps(obj, *args, cls=TypelessSONEncoder, ensure_ascii=False, **kwargs)