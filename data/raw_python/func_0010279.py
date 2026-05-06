def add_tag(self, new_tags):
        """Add a tag to existing device tags. This method will not add a duplicate, if already in the list.

        :param new_tags: the tag(s) to be added. new_tags can be a comma-separated string or list
        """

        tags = self.get_tags()
        orig_tag_cnt = len(tags)
        # print("self.get_tags() {}".format(tags))

        if isinstance(new_tags, six.string_types):
            new_tags = new_tags.split(',')
            # print("spliting tags :: {}".format(new_tags))

        for tag in new_tags:
            if not tag in tags:
                tags.append(tag.strip())

        if len(tags) > orig_tag_cnt:
            xml_tags = escape(",".join(tags))
            post_data = TAGS_TEMPLATE.format(connectware_id=self.get_connectware_id(),
                                             tags=xml_tags)
            self._conn.put('/ws/DeviceCore', post_data)

            # Invalidate cache
            self._device_json = None