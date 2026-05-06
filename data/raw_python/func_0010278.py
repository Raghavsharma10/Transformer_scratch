def add_to_group(self, group_path):
        """Add a device to a group, if the group doesn't exist it is created

        :param group_path: Path or "name" of the group
        """

        if self.get_group_path() != group_path:
            post_data = ADD_GROUP_TEMPLATE.format(connectware_id=self.get_connectware_id(),
                                                  group_path=group_path)
            self._conn.put('/ws/DeviceCore', post_data)

            # Invalidate cache
            self._device_json = None