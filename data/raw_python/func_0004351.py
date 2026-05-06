def delete(self, ids):
        """
        Method to delete pool's by their ids

        :param ids: Identifiers of pool's
        :return: None
        """
        url = build_uri_with_ids('api/v3/pool/%s/', ids)

        return super(ApiPool, self).delete(url)