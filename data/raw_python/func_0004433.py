def delete(self, ids):
        """
        Method to delete vip's by their id's

        :param ids: Identifiers of vip's
        :return: None
        """
        url = build_uri_with_ids('api/v3/vip-request/%s/', ids)

        return super(ApiVipRequest, self).delete(url)