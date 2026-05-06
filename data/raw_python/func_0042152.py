def get_filter_form(self, **kwargs):
        """
        If there is a filter_form, initializes that
        form with the contents of request.GET and
        returns it.
        """

        form = None
        if self.filter_form:
            form = self.filter_form(self.request.GET)
        elif self.model and hasattr(self.model._meta, '_is_view'):
            form = VersionFilterForm(self.request.GET)
        return form