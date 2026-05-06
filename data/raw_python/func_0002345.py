def parent_site(self, site):
        """
        Filter to the given site, only give content relevant for that site.
        """
        # Avoid auto filter if site is already set.
        self._parent_site = site

        if sharedcontent_appsettings.FLUENT_SHARED_CONTENT_ENABLE_CROSS_SITE:
            # Allow content to be shared between all sites:
            return self.filter(Q(parent_site=site) | Q(is_cross_site=True))
        else:
            return self.filter(parent_site=site)