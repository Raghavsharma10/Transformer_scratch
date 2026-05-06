def set_annotations(cls, __fn, *notes, **keyword_notes):
        """Set the annotations on the given callable."""
        if hasattr(__fn, '__func__'):
            __fn = __fn.__func__
        if hasattr(__fn, '__notes__'):
            msg = 'callable already has notes: {!r}'
            raise AttributeError(msg.format(__fn))
        __fn.__notes__ = (notes, keyword_notes)