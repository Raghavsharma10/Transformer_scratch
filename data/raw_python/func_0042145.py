def get_filter(self, **filter_kwargs):
        """
        Returns a list of Q objects that can be passed
        to an queryset for filtering.

        Default implementation returns a Q
        object for `base_filter_kwargs` and any
        passed in keyword arguments.
        """
        filter_kwargs.update(self.base_filter_kwargs)
        if filter_kwargs:
            return [models.Q(**filter_kwargs)]
        return []