def get_kwargs_for_view(self, name):
        """
        Returns the full list of keyword arguments
        for the given view name as a dictionary.
        First the default_kwargs dictionary is copied.
        Then it is updated with the any of the 'view values'
        that can be specified directly on this instance. IE: models.
        Then that dictionary is updated with the values
        particular to this view names from the FOO_kwargs dictionary.
        """

        data = dict(self.default_kwargs)
        for k in self.view_attributes:
            if hasattr(self, k):
                data[k] = getattr(self, k)
        data.update(getattr(self, '%s_kwargs' % name, {}))
        return data