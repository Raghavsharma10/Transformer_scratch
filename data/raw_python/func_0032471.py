def assign(self, droplet_id):
        """
        Assign the floating IP to a droplet

        :param droplet_id: the droplet to assign the floating IP to as either
            an ID or a `Droplet` object
        :type droplet_id: integer or `Droplet`
        :return: an `Action` representing the in-progress operation on the
            floating IP
        :rtype: Action
        :raises DOAPIError: if the API endpoint replies with an error
        """
        if isinstance(droplet_id, Droplet):
            droplet_id = droplet_id.id
        return self.act(type='assign', droplet_id=droplet_id)