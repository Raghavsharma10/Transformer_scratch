def to_link(self):
        """
        Returns a link for the resource.
        """

        link_type = self.link_type if self.type == 'Link' else self.type

        return Link({'sys': {'linkType': link_type, 'id': self.sys.get('id')}}, client=self._client)