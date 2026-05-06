def _single_site(self):
        """
        Make sure the queryset is filtered on a parent site, if that didn't happen already.
        """
        if settings.FLUENTCMS_EMAILTEMPLATES_FILTER_SITE_ID and self._parent_site is None:
            return self.parent_site(settings.SITE_ID)
        else:
            return self