def handle_exception(exc_info=None, source_hint=None, tb_override=_NO):
    """Exception handling helper.  This is used internally to either raise
    rewritten exceptions or return a rendered traceback for the template.
    """

    global _make_traceback
    if exc_info is None:  # pragma: no cover
        exc_info = sys.exc_info()

    # the debugging module is imported when it's used for the first time.
    # we're doing a lot of stuff there and for applications that do not
    # get any exceptions in template rendering there is no need to load
    # all of that.
    if _make_traceback is None:
        from .runtime.debug import make_traceback as _make_traceback

    exc_type, exc_value, tb = exc_info
    if tb_override is not _NO:  # pragma: no cover
        tb = tb_override

    traceback = _make_traceback((exc_type, exc_value, tb), source_hint)
    exc_type, exc_value, tb = traceback.standard_exc_info
    reraise(exc_type, exc_value, tb)