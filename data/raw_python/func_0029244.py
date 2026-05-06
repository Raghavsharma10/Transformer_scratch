def maybe_dotted(module, throw=True):
    """ If ``module`` is a dotted string pointing to the module,
    imports and returns the module object.
    """
    try:
        return Configurator().maybe_dotted(module)
    except ImportError as e:
        err = '%s not found. %s' % (module, e)
        if throw:
            raise ImportError(err)
        else:
            log.error(err)
            return None