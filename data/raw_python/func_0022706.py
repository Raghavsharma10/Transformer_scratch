def sys_info(fname=None, overwrite=False):
    """Get relevant system and debugging information

    Parameters
    ----------
    fname : str | None
        Filename to dump info to. Use None to simply print.
    overwrite : bool
        If True, overwrite file (if it exists).

    Returns
    -------
    out : str
        The system information as a string.
    """
    if fname is not None and op.isfile(fname) and not overwrite:
        raise IOError('file exists, use overwrite=True to overwrite')

    out = ''
    try:
        # Nest all imports here to avoid any circular imports
        from ..app import use_app, Canvas
        from ..app.backends import BACKEND_NAMES
        from ..gloo import gl
        from ..testing import has_backend
        # get default app
        with use_log_level('warning'):
            app = use_app(call_reuse=False)  # suppress messages
        out += 'Platform: %s\n' % platform.platform()
        out += 'Python:   %s\n' % str(sys.version).replace('\n', ' ')
        out += 'Backend:  %s\n' % app.backend_name
        for backend in BACKEND_NAMES:
            if backend.startswith('ipynb_'):
                continue
            with use_log_level('warning', print_msg=False):
                which = has_backend(backend, out=['which'])[1]
            out += '{0:<9} {1}\n'.format(backend + ':', which)
        out += '\n'
        # We need an OpenGL context to get GL info
        canvas = Canvas('Test', (10, 10), show=False, app=app)
        canvas._backend._vispy_set_current()
        out += 'GL version:  %r\n' % (gl.glGetParameter(gl.GL_VERSION),)
        x_ = gl.GL_MAX_TEXTURE_SIZE
        out += 'MAX_TEXTURE_SIZE: %r\n' % (gl.glGetParameter(x_),)
        out += 'Extensions: %r\n' % (gl.glGetParameter(gl.GL_EXTENSIONS),)
        canvas.close()
    except Exception:  # don't stop printing info
        out += '\nInfo-gathering error:\n%s' % traceback.format_exc()
        pass
    if fname is not None:
        with open(fname, 'w') as fid:
            fid.write(out)
    return out