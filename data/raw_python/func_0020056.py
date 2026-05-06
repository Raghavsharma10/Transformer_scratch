def get_serializer(name, **options):
    '''Retrieve a serializer register as *name*. If the serializer is not
available a ``ValueError`` exception will raise.
A common usage pattern::

    qs = MyModel.objects.query().sort_by('id')
    s = odm.get_serializer('json')
    s.dump(qs)
'''
    if name in _serializers:
        serializer = _serializers[name]
        return serializer(**options)
    else:
        raise ValueError('Unknown serializer {0}.'.format(name))