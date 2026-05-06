def delete(self, ids):
        """
        Method to delete neighbors by their id's

        :param ids: Identifiers of neighbors
        :return: None
        """
        url = build_uri_with_ids('api/v4/neighbor/%s/', ids)
        return super(ApiV4Neighbor, self).delete(url)