def _getViewerPrivateApplication(self):
        """
        Get the L{PrivateApplication} object for the logged-in user who is
        viewing this resource, as indicated by its C{username} attribute.

        This is highly problematic because it precludes the possibility of
        separating the stores of the viewer and the viewee into separate
        processes, and it is only here until we can get rid of it.  The reason
        it remains is that some application code still imports things which
        subclass L{PublicAthenaLivePage} and L{PublicPage} and uses them with
        usernames specified.  See ticket #2702 for progress on this goal.

        However, Mantissa itself will no longer set this class's username
        attribute to anything other than None, because authenticated users'
        pages will be generated using
        L{xmantissa.webapp._AuthenticatedWebViewer}.  This method is used only
        to render content in the shell template, and those classes have a direct
        reference to the requisite object.

        @rtype: L{PrivateApplication}
        """
        ls = self.store.findUnique(userbase.LoginSystem)
        substore = ls.accountByAddress(*self.username.split('@')).avatars.open()
        from xmantissa.webapp import PrivateApplication
        return substore.findUnique(PrivateApplication)