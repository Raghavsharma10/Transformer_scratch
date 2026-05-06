def _check_pyopengl_3D():
    """Helper to ensure users have OpenGL for 3D texture support (for now)"""
    global USE_TEX_3D
    USE_TEX_3D = True
    try:
        import OpenGL.GL as _gl
    except ImportError:
        raise ImportError('PyOpenGL is required for 3D texture support')
    return _gl