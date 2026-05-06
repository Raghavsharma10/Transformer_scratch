def admin_stats(self, option):
        """This is a simple way to get statistics about your system.

        :param str option: (required), accepted values: ('all', 'repos',
            'hooks', 'pages', 'orgs', 'users', 'pulls', 'issues',
            'milestones', 'gists', 'comments')
        :returns: dict
        """
        stats = {}
        if option.lower() in ('all', 'repos', 'hooks', 'pages', 'orgs',
                              'users', 'pulls', 'issues', 'milestones',
                              'gists', 'comments'):
            url = self._build_url('enterprise', 'stats', option.lower())
            stats = self._json(self._get(url), 200)
        return stats