def _parse_ret(func, variables, annotations=None):
    """Parse func's return annotation and return either None, a variable,
    or a tuple of variables.

    NOTE:
      * _parse_ret() also notifies variables about will-writes.
      * A variable can be written multiple times per return annotation.
    """
    anno = (annotations or func.__annotations__).get('return')
    if anno is None:
        return None
    elif isinstance(anno, str):
        writeto = variables[anno]
        writeto.notify_will_write()
        return writeto
    elif (isinstance(anno, tuple) and
          all(isinstance(name, str) for name in anno)):
        writeto = tuple(variables[name] for name in anno)
        for var in writeto:
            var.notify_will_write()
        return writeto
    # Be very strict about annotation format for now.
    raise StartupError(
        'cannot parse return annotation %r for %r' % (anno, func))