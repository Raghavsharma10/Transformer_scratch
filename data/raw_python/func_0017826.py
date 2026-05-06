def logs(self, container, tail='all', follow=False, timestamps=False):
        """
        :param container: 'web', 'solr' or 'postgres'
        :param tail: number of lines to show
        :param follow: True to return generator instead of list
        :param timestamps: True to include timestamps
        """
        return container_logs(
            self._get_container_name(container),
            tail,
            follow,
            timestamps)