def get_urls(self):
        """
        Inject extra action URLs.
        """
        urls = []

        for action in self.get_extra_actions():
            regex = r'^{}/$'.format(self._get_action_href(action))
            view = self.admin_site.admin_view(action)
            urls.append(url(regex, view))

        return urls + super(ExtraActionsMixin, self).get_urls()