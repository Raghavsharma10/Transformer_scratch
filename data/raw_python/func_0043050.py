def get(self, request, *args, **kwargs):
        """
        Method for handling GET requests.
        Calls the `render` method with the following
        items in context:

        * **queryset** - Objects to perform action on
        """

        queryset = self.get_selected(request)
        return self.render(request, queryset = queryset)