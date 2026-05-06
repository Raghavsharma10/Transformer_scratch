def resize(self, size, disk=None):
        """
        Resize the droplet

        :param size: a size slug or a `Size` object representing the size to
            resize to
        :type size: string or `Size`
        :param bool disk: Set to `True` for a permanent resize, including
            disk changes
        :return: an `Action` representing the in-progress operation on the
            droplet
        :rtype: Action
        :raises DOAPIError: if the API endpoint replies with an error
        """
        if isinstance(size, Size):
            size = size.slug
        opts = {"disk": disk} if disk is not None else {}
        return self.act(type='resize', size=size, **opts)