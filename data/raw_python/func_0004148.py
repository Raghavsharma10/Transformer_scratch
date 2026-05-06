def delete(self, ids):
        """
        Method to delete network-ipv4's by their ids

        :param ids: Identifiers of network-ipv4's
        :return: None
        """
        url = build_uri_with_ids('api/v3/networkv4/%s/', ids)

        return super(ApiNetworkIPv4, self).delete(url)