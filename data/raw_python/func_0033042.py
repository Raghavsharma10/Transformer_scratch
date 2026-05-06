def fetch_all_snapshots(self):
        r"""
        Returns a generator that yields all of the snapshot images created from
        the droplet

        :rtype: generator of `Image`\ s
        :raises DOAPIError: if the API endpoint replies with an error
        """
        api = self.doapi_manager
        for obj in api.paginate(self.url + '/snapshots', 'snapshots'):
            yield Image(obj, doapi_manager=api)