def delete(self, ids):
        """
        Method to delete ipv6's by their ids

        :param ids: Identifiers of ipv6's
        :return: None
        """
        url = build_uri_with_ids('api/v4/ipv6/%s/', ids)
        return super(ApiV4IPv6, self).delete(url)