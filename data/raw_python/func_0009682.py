def create(self, name, description=None, image_url=None, share=None, **kwargs):
        """Create a new group.

        Note that, although possible, there may be issues when not using an
        image URL from GroupMe's image service.

        :param str name: group name (140 characters maximum)
        :param str description: short description (255 characters maximum)
        :param str image_url: GroupMe image service URL
        :param bool share: whether to generate a share URL
        :return: a new group
        :rtype: :class:`~groupy.api.groups.Group`
        """
        payload = {
            'name': name,
            'description': description,
            'image_url': image_url,
            'share': share,
        }
        payload.update(kwargs)
        response = self.session.post(self.url, json=payload)
        return Group(self, **response.data)