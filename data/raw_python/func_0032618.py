def render_settingsLink(self, ctx, data):
        """
        Add the URL of the settings page to the given tag.

        @see L{xmantissa.webnav.settingsLink}
        """
        return settingsLink(
            self.translator, self.pageComponents.settings, ctx.tag)