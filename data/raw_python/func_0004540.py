def delete(self, ids):
        """
        Method to delete network-ipv6's by their ids

        :param ids: Identifiers of network-ipv6's
        :return: None
        """
        url = build_uri_with_ids('api/v3/networkv6/%s/', ids)

        return super(ApiNetworkIPv6, self).delete(url)