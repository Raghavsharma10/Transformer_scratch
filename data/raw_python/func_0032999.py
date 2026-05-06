def render_settingsLink(self, ctx, data):
        """
        For authenticated users, add the URL of the settings page to the given
        tag.  For unauthenticated users, remove the given tag from the output.
        """
        if self.username is None:
            return ''
        translator = self._getViewerPrivateApplication()
        return settingsLink(
            translator,
            translator.getPageComponents().settings,
            ctx.tag)