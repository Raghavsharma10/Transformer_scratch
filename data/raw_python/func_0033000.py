def render_applicationNavigation(self, ctx, data):
        """
        For authenticated users, add primary application navigation to the
        given tag.  For unauthenticated users, remove the given tag from the
        output.

        @see L{xmantissa.webnav.applicationNavigation}
        """
        if self.username is None:
            return ''
        translator = self._getViewerPrivateApplication()
        return applicationNavigation(
            ctx,
            translator,
            translator.getPageComponents().navigation)