def _stop_timers(canvas):
    """Stop all timers in a canvas."""
    for attr in dir(canvas):
        try:
            attr_obj = getattr(canvas, attr)
        except NotImplementedError:
            # This try/except is needed because canvas.position raises
            # an error (it is not implemented in this backend).
            attr_obj = None
        if isinstance(attr_obj, Timer):
            attr_obj.stop()