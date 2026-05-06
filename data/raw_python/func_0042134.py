def get_render_data(self, **kwargs):
        """
        Because of the way mixin inheritance works
        we can't have a default implementation of
        get_context_data on the this class, so this
        calls that method if available and returns
        the resulting context.
        """
        if hasattr(self, 'get_context_data'):
            data = self.get_context_data(**kwargs)
        else:
            data = kwargs
        return data