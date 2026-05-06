def create_floating_ip(self, droplet_id=None, region=None, **kwargs):
        """
        Create a new floating IP assigned to a droplet or reserved to a region.
        Either ``droplet_id`` or ``region`` must be specified, but not both.

        The returned `FloatingIP` object will represent the IP at the moment of
        creation; if the IP address is supposed to be assigned to a droplet,
        the assignment may not have been completed at the time the object is
        returned.  To wait for the assignment to complete, use the
        `FloatingIP`'s :meth:`~FloatingIP.wait_for_action` method.

        :param droplet_id: the droplet to assign the floating IP to as either
            an ID or a `Droplet` object
        :type droplet_id: integer or `Droplet`
        :param region: the region to reserve the floating IP to as either a
            slug or a `Region` object
        :type region: string or `Region`
        :param kwargs: additional fields to include in the API request
        :return: the new floating IP
        :rtype: FloatingIP
        :raises TypeError: if both ``droplet_id`` & ``region`` or neither of
            them are defined
        :raises DOAPIError: if the API endpoint replies with an error
        """
        if (droplet_id is None) == (region is None):
            ### TODO: Is TypeError the right type of error?
            raise TypeError('Exactly one of "droplet_id" and "region" must be'
                            ' specified')
        if droplet_id is not None:
            if isinstance(droplet_id, Droplet):
                droplet_id = droplet_id.id
            data = {"droplet_id": droplet_id}
        else:
            if isinstance(region, Region):
                region = region.slug
            data = {"region": region}
        data.update(kwargs)
        return self._floating_ip(self.request('/v2/floating_ips', method='POST',
                                              data=data)["floating_ip"])