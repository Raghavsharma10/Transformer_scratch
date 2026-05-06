def delete(self, ids):
        """
        Method to delete environments vip by their id's.

        :param ids: Identifiers of environments vip
        :return: None
        """
        url = build_uri_with_ids('api/v3/environment-vip/%s/', ids)
        return super(ApiEnvironmentVip, self).delete(url)