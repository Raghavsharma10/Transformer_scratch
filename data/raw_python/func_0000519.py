def is_visible(self):
        """
        Return a boolean if the page is visible in navigation.

        Pages must have show in navigation set. Regular pages must be published (published and
        have a current version - checked with `is_published`), pages with a glitter app associated
        don't need any page versions.
        """
        if self.glitter_app_name:
            visible = self.show_in_navigation
        else:
            visible = self.show_in_navigation and self.is_published

        return visible