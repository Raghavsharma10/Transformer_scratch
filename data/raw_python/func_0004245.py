def delete(self, ids):
        """
        Method to delete Virtual Interfaces by their id's

        :param ids: Identifiers of Virtual Interfaces
        :return: None
        """
        url = build_uri_with_ids('api/v4/virtual-interface/%s/', ids)
        return super(ApiV4VirtualInterface, self).delete(url)