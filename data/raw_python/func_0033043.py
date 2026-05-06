def fetch_all_backups(self):
        r"""
        Returns a generator that yields all of the backup images created from
        the droplet

        :rtype: generator of `Image`\ s
        :raises DOAPIError: if the API endpoint replies with an error
        """
        api = self.doapi_manager
        for obj in api.paginate(self.url + '/backups', 'backups'):
            yield Image(obj, doapi_manager=api)