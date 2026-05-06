def fetch_all_neighbors(self):
        r"""
        Returns a generator that yields all of the droplets running on the same
        physical server as the droplet

        :rtype: generator of `Droplet`\ s
        :raises DOAPIError: if the API endpoint replies with an error
        """
        api = self.doapi_manager
        return map(api._droplet, api.paginate(self.url + '/neighbors',
                                             'droplets'))