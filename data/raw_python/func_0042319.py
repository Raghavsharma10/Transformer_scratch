def b64decode(foo, *args):
    'Only here for consistency with the above.'
    if isinstance(foo, str):
        foo = foo.encode('utf8')
    return base64.b64decode(foo, *args)