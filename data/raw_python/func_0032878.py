def fetch_all_droplets(self, tag_name=None):
        r"""
        Returns a generator that yields all of the droplets belonging to the
        account

        .. versionchanged:: 0.2.0
            ``tag_name`` parameter added

        :param tag_name: if non-`None`, only droplets with the given tag are
            returned
        :type tag_name: string or `Tag`
        :rtype: generator of `Droplet`\ s
        :raises DOAPIError: if the API endpoint replies with an error
        """
        params = {}
        if tag_name is not None:
            params["tag_name"] = str(tag_name)
        return map(self._droplet, self.paginate('/v2/droplets', 'droplets',
                                                params=params))