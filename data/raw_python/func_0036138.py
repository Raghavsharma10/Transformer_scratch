def get_volumes_for_instance(self, arg, device=None):
        """
        Return all EC2 Volume objects attached to ``arg`` instance name or ID.

        May specify ``device`` to limit to the (single) volume attached as that
        device.
        """
        instance = self.get(arg)
        filters = {'attachment.instance-id': instance.id}
        if device is not None:
            filters['attachment.device'] = device
        return self.get_all_volumes(filters=filters)