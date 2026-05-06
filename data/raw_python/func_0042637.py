def get_object(self):
        """
        Get the object we are working with. Makes sure
        get_queryset is called even when in add mode.
        """

        if not self.force_add and self.kwargs.get(self.slug_url_kwarg, None):
            return super(FormView, self).get_object()
        else:
            self.queryset = self.get_queryset()

        return None