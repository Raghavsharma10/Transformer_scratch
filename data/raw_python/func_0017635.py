def root(self, request, url):
        """
        Handles main URL routing for the databrowse app.

        `url` is the remainder of the URL -- e.g. 'objects/3'.
        """
        # Delegate to the appropriate method, based on the URL.
        if url is None:
            return self.main_view(request)
        try:
            plugin_name, rest_of_url = url.split('/', 1)
        except ValueError:  # need more than 1 value to unpack
            plugin_name, rest_of_url = url, None
        try:
            plugin = self.plugins[plugin_name]
        except KeyError:
            raise http.Http404('A plugin with the requested name '
                               'does not exist.')
        return plugin.model_view(request, self, rest_of_url)