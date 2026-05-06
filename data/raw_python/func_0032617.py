def render_startmenu(self, ctx, data):
        """
        Add start-menu style navigation to the given tag.

        @see {xmantissa.webnav.startMenu}
        """
        return startMenu(
            self.translator, self.pageComponents.navigation, ctx.tag)