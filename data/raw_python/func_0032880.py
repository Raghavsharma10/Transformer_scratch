def create_multiple_droplets(self, names, image, size, region,
                                 ssh_keys=None, backups=None, ipv6=None,
                                 private_networking=None, user_data=None,
                                 **kwargs):
        r"""
        Create multiple new droplets at once with the same image, size, etc.,
        differing only in name.  All fields other than ``names``, ``image``,
        ``size``, and ``region`` are optional and will be omitted from the API
        request if not specified.

        The returned `Droplet` objects will represent the droplets at the
        moment of creation; the actual droplets may not be active yet and may
        not have even been assigned IP addresses.  To wait for the droplets to
        activate, use their :meth:`~Droplet.wait` method or `wait_droplets`.

        :param names: the names for the new droplets
        :type names: list of strings
        :param image: the image ID, slug, or `Image` object representing the
            base image to use for the droplets
        :type image: integer, string, or `Image`
        :param size: the slug or `Size` object representing the size of the new
            droplets
        :type size: string or `Size`
        :param region: the slug or `Region` object representing the region in
            which to create the droplets
        :type region: string or `Region`
        :param iterable ssh_keys: an iterable of SSH key resource IDs, SSH key
            fingerprints, and/or `SSHKey` objects specifying the public keys to
            add to the new droplets' :file:`/root/.ssh/authorized_keys` files
        :param bool backups: whether to enable automatic backups on the new
            droplets
        :param bool ipv6: whether to enable IPv6 on the new droplets
        :param bool private_networking: whether to enable private networking
            for the new droplets
        :param str user_data: a string of user data/metadata for the droplets
        :param kwargs: additional fields to include in the API request
        :return: the new droplet resources
        :rtype: list of `Droplet`\ s
        :raises DOAPIError: if the API endpoint replies with an error
        """
        data = {
            "names": names,
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
        return list(map(self._droplet, self.request('/v2/droplets',
                                                    method='POST',
                                                    data=data)["droplets"]))