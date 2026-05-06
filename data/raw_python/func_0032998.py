def render_startmenu(self, ctx, data):
        """
        For authenticated users, add the start-menu style navigation to the
        given tag.  For unauthenticated users, remove the given tag from the
        output.

        @see L{xmantissa.webnav.startMenu}
        """
        if self.username is None:
            return ''
        translator = self._getViewerPrivateApplication()
        pageComponents = translator.getPageComponents()
        return startMenu(translator, pageComponents.navigation, ctx.tag)