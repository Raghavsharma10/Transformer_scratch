def _is_readable(self, obj):
        """Check if the argument is a readable file-like object."""
        try:
            read = getattr(obj, 'read')
        except AttributeError:
            return False
        else:
            return is_method(read, max_arity=1)