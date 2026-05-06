def forget_canvas(canvas):
    """ Forget about the given canvas. Used by the canvas when closed.
    """
    cc = [c() for c in canvasses if c() is not None]
    while canvas in cc:
        cc.remove(canvas)
    canvasses[:] = [weakref.ref(c) for c in cc]