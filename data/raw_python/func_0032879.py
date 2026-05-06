def create_droplet(self, name, image, size, region, ssh_keys=None,
                       backups=None, ipv6=None, private_networking=None,
                       user_data=None, **kwargs):
        """
        Create a new droplet.  All fields other than ``name``, ``image``,
        ``size``, and ``region`` are optional and will be omitted from the API
        request if not specified.

        The returned `Droplet` object will represent the droplet at the moment
        of creation; the actual droplet may not be active yet and may not have
        even been assigned an IP address.  To wait for the droplet to activate,
        use the `Droplet`'s :meth:`~Droplet.wait` method.

        :param str name: a name for the droplet
        :param image: the image ID, slug, or `Image` object representing the
            base image to use for the droplet
        :type image: integer, string, or `Image`
        :param size: the slug or `Size` object representing the size of the new
            droplet
        :type size: string or `Size`
        :param region: the slug or `Region` object representing the region in
            which to create the droplet
        :type region: string or `Region`
        :param iterable ssh_keys: an iterable of SSH key resource IDs, SSH key
            fingerprints, and/or `SSHKey` objects specifying the public keys to
            add to the new droplet's :file:`/root/.ssh/authorized_keys` file
        :param bool backups: whether to enable automatic backups on the new
            droplet
        :param bool ipv6: whether to enable IPv6 on the new droplet
        :param bool private_networking: whether to enable private networking
            for the new droplet
        :param str user_data: a string of user data/metadata for the droplet
        :param kwargs: additional fields to include in the API request
        :return: the new droplet resource
        :rtype: Droplet
        :raises DOAPIError: if the API endpoint replies with an error
        """
        data = {
            "name": name,
            "image": image.id if isinstance(image, Image) else image,
            "size": str(size),
            "region": str(region),
        }
        if ssh_keys is not None:
            data["ssh_keys"] = [k._id if isinstance(k, SSHKey) else k
                                      for k in ssh_keys]
        if backups is not None:
            data["backups"] = backups
        if ipv6 is not None:
            data["ipv6"] = ipv6
        if private_networking is not None:
            data["private_networking"] = private_networking
        if user_data is not None:
            data["user_data"] = user_data
        data.update(kwargs)
        return self._droplet(self.request('/v2/droplets', method='POST',
                                          data=data)["droplet"])