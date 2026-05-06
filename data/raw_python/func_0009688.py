def update(self, name=None, description=None, image_url=None,
               office_mode=None, share=None, **kwargs):
        """Update the details of the group.

        :param str name: group name (140 characters maximum)
        :param str description: short description (255 characters maximum)
        :param str image_url: GroupMe image service URL
        :param bool office_mode: (undocumented)
        :param bool share: whether to generate a share URL
        :return: an updated group
        :rtype: :class:`~groupy.api.groups.Group`
        """
        # note we default to the current values for name and office_mode as a
        # work-around for issues with the group update endpoint
        if name is None:
            name = self.name
        if office_mode is None:
            office_mode = self.office_mode
        return self.manager.update(id=self.id, name=name, description=description,
                                   image_url=image_url, office_mode=office_mode,
                                   share=share, **kwargs)