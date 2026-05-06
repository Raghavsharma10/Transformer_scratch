def get_annotations(cls, __fn):
        """Get the annotations of a given callable."""
        if hasattr(__fn, '__func__'):
            __fn = __fn.__func__
        if hasattr(__fn, '__notes__'):
            return __fn.__notes__
        raise AttributeError('{!r} does not have annotations'.format(__fn))