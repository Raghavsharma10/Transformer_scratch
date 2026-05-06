def data_offerings(self, ctx, data):
        """
        Generate a list of installed offerings.

        @return: a generator of dictionaries mapping 'name' to the name of an
        offering installed on the store.
        """
        for io in self.original.store.query(offering.InstalledOffering):
            pp = ixmantissa.IPublicPage(io.application, None)
            if pp is not None and getattr(pp, 'index', True):
                warn("Use the sharing system to provide public pages,"
                     " not IPublicPage",
                     category=DeprecationWarning,
                     stacklevel=2)
                yield {'name': io.offeringName}
            else:
                s = io.application.open()
                try:
                    pp = getEveryoneRole(s).getShare(getDefaultShareID(s))
                    yield {'name': io.offeringName}
                except NoSuchShare:
                    continue