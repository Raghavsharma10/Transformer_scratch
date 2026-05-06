def _is_writable(self, obj):
        """Check if the argument is a writable file-like object."""
        try:
            write = getattr(obj, 'write')
        except AttributeError:
            return False
        else:
            return is_method(write, min_arity=1, max_arity=1)