def django_api(
            self,
            name,
            introduced_at,
            undocumented=False,
            deprecated_at=None,
            title=None,
            **options):
        """Add a django API handler to the service.

        :param name: This is the name of the django url to use.

        The 'methods' paramater can be supplied as normal, you can also user
        the @api.handler decorator to link this API to its handler.

        """
        from acceptable.djangoutil import DjangoAPI
        location = get_callsite_location()
        api = DjangoAPI(
            self,
            name,
            introduced_at,
            options,
            location=location,
            undocumented=undocumented,
            deprecated_at=deprecated_at,
            title=title,
        )
        self.metadata.register_api(self.name, self.group, api)
        return api