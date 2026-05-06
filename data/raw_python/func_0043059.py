def get(self, request, *args, **kwargs):
        """
        Method for handling GET requests. Passes the
        following arguments to the context:

        * **obj** - The object to publish
        * **done_url** - The result of the `get_done_url` method
        """

        self.object = self.get_object()
        return self.render(request, obj=self.object,
                           done_url=self.get_done_url())