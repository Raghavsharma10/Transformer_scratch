def register(self, slug, bundle, order=1, title=None):
        """
        Registers the bundle for a certain slug.

        If a slug is already registered, this will raise AlreadyRegistered.

        :param slug: The slug to register.
        :param bundle: The bundle instance being registered.
        :param order: An integer that controls where this bundle's \
        dashboard links appear in relation to others.
        """

        if slug in self._registry:
            raise AlreadyRegistered('The url %s is already registered' % slug)

        # Instantiate the admin class to save in the registry.
        self._registry[slug] = bundle
        self._order[slug] = order
        if title:
            self._titles[slug] = title
        bundle.set_admin_site(self)