def register_pickle():
    """The fastest serialization method, but restricts
    you to python clients."""
    import cPickle
    registry.register('pickle', cPickle.dumps, cPickle.loads,
                      content_type='application/x-python-serialize',
                      content_encoding='binary')