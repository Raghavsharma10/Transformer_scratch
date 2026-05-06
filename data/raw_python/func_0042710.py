def unregister(self, slug):
        """
        Unregisters the given url.

        If a slug isn't already registered, this will raise NotRegistered.
        """

        if slug not in self._registry:
            raise NotRegistered('The slug %s is not registered' % slug)
        bundle = self._registry[slug]
        if bundle._meta.model and bundle._meta.primary_model_bundle:
            self.unregister_model(bundle._meta.model)

        del self._registry[slug]
        del self._order[slug]