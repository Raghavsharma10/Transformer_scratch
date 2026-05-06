def get(self, *args, **kwargs):
        """See if this view was called with a specified category."""
        self.initial = {"category_name":  kwargs.get('category_name', None)}
        return super(CategoryFormView, self).get(*args, **kwargs)