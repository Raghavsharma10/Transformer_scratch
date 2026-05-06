def delete(self, ids):
        """
        Method to delete ipv4's by their ids

        :param ids: Identifiers of ipv4's
        :return: None
        """
        url = build_uri_with_ids('api/v4/ipv4/%s/', ids)

        return super(ApiV4IPv4, self).delete(url)