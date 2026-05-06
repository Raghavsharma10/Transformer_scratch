def check_error(when='periodic check'):
    """ Check this from time to time to detect GL errors.

    Parameters
    ----------
    when : str
        Shown in the exception to help the developer determine when
        this check was done.
    """

    errors = []
    while True:
        err = glGetError()
        if err == GL_NO_ERROR or (errors and err == errors[-1]):
            break
        errors.append(err)
    if errors:
        msg = ', '.join([repr(ENUM_MAP.get(e, e)) for e in errors])
        err = RuntimeError('OpenGL got errors (%s): %s' % (when, msg))
        err.errors = errors
        err.err = errors[-1]  # pyopengl compat
        raise err