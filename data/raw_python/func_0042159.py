def get_context_data(self, **kwargs):
        """
        Get the context for this view. Adds the following values:

        * **query_string** - The querystring minus the current page.
        * **action_links** - The results of the `get_actions` method.
        """

        origin_qs = self._get_query_string(self.request, False)
        context = {
            'query_string': self._get_query_string(self.request),
            'origin_qs': self.request.path + origin_qs,
            'origin_var': self.ORIGIN_ARGUMENT,
            'action_links': self.get_actions()
        }
        context.update(kwargs)

        return context