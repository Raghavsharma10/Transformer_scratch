def get(self, request, *args, **kwargs):
        """
        Method for handling GET requests. Passes the
        following arguments to the context:

        * **versions** - The versions available for this object.\
        These will be instances of the inner version class, and \
        will not have access to the fields on the base model.
        * **done_url** - The result of the `get_done_url` method.
        """
        versions = self._get_versions()
        return self.render(request, obj=self.object, versions=versions,
                           done_url=self.get_done_url())