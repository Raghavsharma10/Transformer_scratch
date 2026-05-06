def delete(self, *args, **kwargs):
        """
        Delete clonable relations first, since they may be
        objects that wouldn't otherwise be deleted.

        Calls super to actually delete the object.
        """
        skip_reverses = kwargs.pop('skip_reverses', False)
        if not skip_reverses:
            self._delete_reverses()

        return super(Cloneable, self).delete(*args, **kwargs)