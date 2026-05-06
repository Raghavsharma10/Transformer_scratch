def delete(self, ids):
        """
        Method to delete asns by their id's

        :param ids: Identifiers of asns
        :return: None
        """
        url = build_uri_with_ids('api/v4/as/%s/', ids)
        return super(ApiV4As, self).delete(url)