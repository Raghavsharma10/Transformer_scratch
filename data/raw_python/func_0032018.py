def set_tags(self, tags):
        """For every known tag, set the appropriate attribute.

        Known tags are:

            :color: The user color
            :emotes: A list of emotes
            :subscriber: True, if subscriber
            :turbo: True, if turbo user
            :user_type: None, mod, staff, global_mod, admin

        :param tags: a list of tags
        :type tags: :class:`list` of :class:`Tag` | None
        :returns: None
        :rtype: None
        :raises: None
        """
        if tags is None:
            return
        attrmap = {'color': 'color', 'emotes': 'emotes',
                   'subscriber': 'subscriber',
                   'turbo': 'turbo', 'user-type': 'user_type'}
        for t in tags:
            attr = attrmap.get(t.name)
            if not attr:
                continue
            else:
                setattr(self, attr, t.value)