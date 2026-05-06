def update(self, id, name=None, description=None, image_url=None,
               office_mode=None, share=None, **kwargs):
        """Update the details of a group.

        .. note::

            There are significant bugs in this endpoint!
            1. not providing ``name`` produces 400: "Topic can't be blank"
            2. not providing ``office_mode`` produces 500: "sql: Scan error on
            column index 14: sql/driver: couldn't convert <nil> (<nil>) into
            type bool"

            Note that these issues are "handled" automatically when calling
            update on a :class:`~groupy.api.groups.Group` object.

        :param str id: group ID
        :param str name: group name (140 characters maximum)
        :param str description: short description (255 characters maximum)
        :param str image_url: GroupMe image service URL
        :param bool office_mode: (undocumented)
        :param bool share: whether to generate a share URL
        :return: an updated group
        :rtype: :class:`~groupy.api.groups.Group`
        """
        path = '{}/update'.format(id)
        url = utils.urljoin(self.url, path)
        payload = {
            'name': name,
            'description': description,
            'image_url': image_url,
            'office_mode': office_mode,
            'share': share,
        }
        payload.update(kwargs)
        response = self.session.post(url, json=payload)
        return Group(self, **response.data)