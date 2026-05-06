def register_msgpack():
    """See http://msgpack.sourceforge.net/"""
    try:
        import msgpack
        registry.register('msgpack', msgpack.packs, msgpack.unpacks,
                content_type='application/x-msgpack',
                content_encoding='binary')
    except ImportError:

        def not_available(*args, **kwargs):
            """In case a client receives a msgpack message, but yaml
            isn't installed."""
            raise SerializerNotInstalled(
                    "No decoder installed for msgpack. "
                    "Install the msgpack library")
        registry.register('msgpack', None, not_available,
                          'application/x-msgpack')