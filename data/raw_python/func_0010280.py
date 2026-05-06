def remove_tag(self, tag):
        """Remove tag from existing device tags

        :param tag: the tag to be removed from the list

        :raises ValueError: If tag does not exist in list
        """

        tags = self.get_tags()
        tags.remove(tag)

        post_data = TAGS_TEMPLATE.format(connectware_id=self.get_connectware_id(),
                                         tags=escape(",".join(tags)))
        self._conn.put('/ws/DeviceCore', post_data)

        # Invalidate cache
        self._device_json = None