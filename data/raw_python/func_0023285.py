def use(app=None, gl=None):
    """ Set the usage options for vispy

    Specify what app backend and GL backend to use.

    Parameters
    ----------
    app : str
        The app backend to use (case insensitive). Standard backends:
            * 'PyQt4': use Qt widget toolkit via PyQt4.
            * 'PyQt5': use Qt widget toolkit via PyQt5.
            * 'PySide': use Qt widget toolkit via PySide.
            * 'PyGlet': use Pyglet backend.
            * 'Glfw': use Glfw backend (successor of Glut). Widely available
              on Linux.
            * 'SDL2': use SDL v2 backend.
            * 'osmesa': Use OSMesa backend
        Additional backends:
            * 'ipynb_vnc': render in the IPython notebook via a VNC approach
              (experimental)
    gl : str
        The gl backend to use (case insensitive). Options are:
            * 'gl2': use Vispy's desktop OpenGL API.
            * 'pyopengl2': use PyOpenGL's desktop OpenGL API. Mostly for
              testing.
            * 'es2': (TO COME) use real OpenGL ES 2.0 on Windows via Angle.
              Availability of ES 2.0 is larger for Windows, since it relies
              on DirectX.
            * 'gl+': use the full OpenGL functionality available on
              your system (via PyOpenGL).

    Notes
    -----
    If the app option is given, ``vispy.app.use_app()`` is called. If
    the gl option is given, ``vispy.gloo.use_gl()`` is called.

    If an app backend name is provided, and that backend could not be
    loaded, an error is raised.

    If no backend name is provided, Vispy will first check if the GUI
    toolkit corresponding to each backend is already imported, and try
    that backend first. If this is unsuccessful, it will try the
    'default_backend' provided in the vispy config. If still not
    succesful, it will try each backend in a predetermined order.

    See Also
    --------
    vispy.app.use_app
    vispy.gloo.gl.use_gl
    """
    if app is None and gl is None:
        raise TypeError('Must specify at least one of "app" or "gl".')

    # Example for future. This wont work (yet).
    if app == 'ipynb_webgl':
        app = 'headless'
        gl = 'webgl'

    if app == 'osmesa':
        from ..util.osmesa_gl import fix_osmesa_gl_lib
        fix_osmesa_gl_lib()
        if gl is not None:
            raise ValueError("Do not specify gl when using osmesa")

    # Apply now
    if gl:
        from .. import gloo, config
        config['gl_backend'] = gl
        gloo.gl.use_gl(gl)
    if app:
        from ..app import use_app
        use_app(app)