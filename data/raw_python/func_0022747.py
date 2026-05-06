def set_current_canvas(canvas):
    """ Make a canvas active. Used primarily by the canvas itself.
    """
    # Notify glir 
    canvas.context._do_CURRENT_command = True
    # Try to be quick
    if canvasses and canvasses[-1]() is canvas:
        return
    # Make this the current
    cc = [c() for c in canvasses if c() is not None]
    while canvas in cc:
        cc.remove(canvas)
    cc.append(canvas)
    canvasses[:] = [weakref.ref(c) for c in cc]