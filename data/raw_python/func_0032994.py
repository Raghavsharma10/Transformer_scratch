def _wrapNavFrag(self, frag, useAthena):
        """
        Wrap the given L{INavigableFragment} in the appropriate type of
        L{_PublicPageMixin}.
        """
        if useAthena:
            return PublicAthenaLivePage(self._siteStore, frag)
        else:
            return PublicPage(None, self._siteStore, frag, None, None)