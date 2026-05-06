def dispatch(self, request, *args, **kwargs):
        """Adds useful objects to the class."""
        self._add_next_and_user(request)
        return super(UpdateImageView, self).dispatch(request, *args, **kwargs)