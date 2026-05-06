def _patch():
    """ Monkey-patch pyopengl to fix a bug in glBufferSubData. """
    import sys
    from OpenGL import GL
    if sys.version_info > (3,):
        buffersubdatafunc = GL.glBufferSubData
        if hasattr(buffersubdatafunc, 'wrapperFunction'):
            buffersubdatafunc = buffersubdatafunc.wrapperFunction
        _m = sys.modules[buffersubdatafunc.__module__]
        _m.long = int
    
    # Fix missing enum
    try:
        from OpenGL.GL.VERSION import GL_2_0
        GL_2_0.GL_OBJECT_SHADER_SOURCE_LENGTH = GL_2_0.GL_SHADER_SOURCE_LENGTH
    except Exception:
        pass