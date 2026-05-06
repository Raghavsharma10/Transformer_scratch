def dispatch(self, *args, **kwargs):
        """This decorator sets this view to have restricted permissions."""
        return super(AnimalDelete, self).dispatch(*args, **kwargs)