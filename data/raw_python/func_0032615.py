def _wrapNavFrag(self, frag, useAthena):
        """
        Wrap the given L{INavigableFragment} in an appropriate
        L{_FragmentWrapperMixin} subclass.
        """
        username = self._privateApplication._getUsername()
        cf = getattr(frag, 'customizeFor', None)
        if cf is not None:
            frag = cf(username)
        if useAthena:
            pageClass = GenericNavigationAthenaPage
        else:
            pageClass = GenericNavigationPage
        return pageClass(self._privateApplication, frag,
                         self._privateApplication.getPageComponents(),
                         username)