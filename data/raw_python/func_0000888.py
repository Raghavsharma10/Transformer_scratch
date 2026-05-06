def api(self,
            url,
            name,
            introduced_at=None,
            undocumented=False,
            deprecated_at=None,
            title=None,
            **options):
        """Add an API to the service.

        :param url: This is the url that the API should be registered at.
        :param name: This is the name of the api, and will be registered with
            flask apps under.

        Other keyword arguments may be used, and they will be passed to the
        flask application when initialised. Of particular interest is the
        'methods' keyword argument, which can be used to specify the HTTP
        method the URL will be added for.
        """
        location = get_callsite_location()
        api = AcceptableAPI(
            self,
            name,
            url,
            introduced_at,
            options,
            undocumented=undocumented,
            deprecated_at=deprecated_at,
            title=title,
            location=location,
        )
        self.metadata.register_api(self.name, self.group, api)
        return api