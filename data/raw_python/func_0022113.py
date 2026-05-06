def get_search_url(self):
        """
        resolve the search url no matter if local or remote.
        :return: url or exception
        """

        if self.is_remote:
            return self.url

        return reverse('search_api', args=[self.slug])