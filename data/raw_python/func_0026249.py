def _get_placeholder_repr(self):
        """Return the placeholder part of matcher's ``__repr__``."""
        placeholder = '...'
        if self.TRANSFORM is not None:
            placeholder = '%s(%s)' % (self.TRANSFORM.__name__, placeholder)
        return placeholder