def create_engine(engine, options=None, defaults=None):
    '''
    Creates an instance of an engine.
    There is a two-stage instantiation process with engines.

    1. ``options``:
        The keyword options to instantiate the engine class
    2. ``defaults``:
        The default configuration for the engine (options often depends on instantiated TTS engine)
    '''
    if engine not in _ENGINE_MAP.keys():
        raise TTSError('Unknown engine %s' % engine)

    options = options or {}
    defaults = defaults or {}
    einst = _ENGINE_MAP[engine](**options)
    einst.configure_default(**defaults)
    return einst