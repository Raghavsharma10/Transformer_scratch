def userBrowser(self, request, tag):
        """
        Render a TDB of local users.
        """
        f = LocalUserBrowserFragment(self.browser)
        f.docFactory = webtheme.getLoader(f.fragmentName)
        f.setFragmentParent(self)
        return f