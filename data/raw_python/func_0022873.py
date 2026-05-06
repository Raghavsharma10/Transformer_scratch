def read_pixels(viewport=None, alpha=True, out_type='unsigned_byte'):
    """Read pixels from the currently selected buffer. 
    
    Under most circumstances, this function reads from the front buffer.
    Unlike all other functions in vispy.gloo, this function directly executes
    an OpenGL command.

    Parameters
    ----------
    viewport : array-like | None
        4-element list of x, y, w, h parameters. If None (default),
        the current GL viewport will be queried and used.
    alpha : bool
        If True (default), the returned array has 4 elements (RGBA).
        If False, it has 3 (RGB).
    out_type : str | dtype
        Can be 'unsigned_byte' or 'float'. Note that this does not
        use casting, but instead determines how values are read from
        the current buffer. Can also be numpy dtypes ``np.uint8``,
        ``np.ubyte``, or ``np.float32``.

    Returns
    -------
    pixels : array
        3D array of pixels in np.uint8 or np.float32 format. 
        The array shape is (h, w, 3) or (h, w, 4), with the top-left corner 
        of the framebuffer at index [0, 0] in the returned array.
    """
    # Check whether the GL context is direct or remote
    context = get_current_canvas().context
    if context.shared.parser.is_remote():
        raise RuntimeError('Cannot use read_pixels() with remote GLIR parser')
    
    finish()  # noqa - finish first, also flushes GLIR commands
    type_dict = {'unsigned_byte': gl.GL_UNSIGNED_BYTE,
                 np.uint8: gl.GL_UNSIGNED_BYTE,
                 'float': gl.GL_FLOAT,
                 np.float32: gl.GL_FLOAT}
    type_ = _check_conversion(out_type, type_dict)
    if viewport is None:
        viewport = gl.glGetParameter(gl.GL_VIEWPORT)
    viewport = np.array(viewport, int)
    if viewport.ndim != 1 or viewport.size != 4:
        raise ValueError('viewport should be 1D 4-element array-like, not %s'
                         % (viewport,))
    x, y, w, h = viewport
    gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)  # PACK, not UNPACK
    fmt = gl.GL_RGBA if alpha else gl.GL_RGB
    im = gl.glReadPixels(x, y, w, h, fmt, type_)
    gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 4)
    # reshape, flip, and return
    if not isinstance(im, np.ndarray):
        np_dtype = np.uint8 if type_ == gl.GL_UNSIGNED_BYTE else np.float32
        im = np.frombuffer(im, np_dtype)

    im.shape = h, w, (4 if alpha else 3)  # RGBA vs RGB
    im = im[::-1, :, :]  # flip the image
    return im