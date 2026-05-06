def delete(self, ids):
        """
        Method to delete vlan's by their ids

        :param ids: Identifiers of vlan's
        :return: None
        """
        url = build_uri_with_ids('api/v3/vlan/%s/', ids)

        return super(ApiVlan, self).delete(url)