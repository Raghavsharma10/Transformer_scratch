def render_applicationNavigation(self, ctx, data):
        """
        Add primary application navigation to the given tag.

        @see L{xmantissa.webnav.applicationNavigation}
        """
        return applicationNavigation(
            ctx, self.translator, self.pageComponents.navigation)