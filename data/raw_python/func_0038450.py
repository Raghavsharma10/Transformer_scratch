def add(self, parent, obj_type, **attributes):
        """ IXN API add command

        @param parent: object parent - object will be created under this parent.
        @param object_type: object type.
        @param attributes: additional attributes.
        @return: STC object reference.
        """

        response = self.post(self.server_url + parent.ref + '/' + obj_type, attributes)
        return self._get_href(response.json())