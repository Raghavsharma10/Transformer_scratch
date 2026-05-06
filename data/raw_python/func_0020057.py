def register_serializer(name, serializer):
    '''\
Register a new serializer to the library.

:parameter name: serializer name (it can override existing serializers).
:parameter serializer: an instance or a derived class of a
    :class:`stdnet.odm.Serializer` class or a callable.
'''
    if not isclass(serializer):
        serializer = serializer.__class__
    _serializers[name] = serializer