def fetch_all_droplet_neighbors(self):
        r"""
        Returns a generator of all sets of multiple droplets that are running
        on the same physical hardware

        :rtype: generator of lists of `Droplet`\ s
        :raises DOAPIError: if the API endpoint replies with an error
        """
        for hood in self.paginate('/v2/reports/droplet_neighbors', 'neighbors'):
            yield list(map(self._droplet, hood))