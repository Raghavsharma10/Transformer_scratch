def dispatch(self, *args, **kwargs):
        """This decorator sets this view to have restricted permissions."""
        return super(BreedingDelete, self).dispatch(*args, **kwargs)