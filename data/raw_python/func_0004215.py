def delete(self, ids):
        """
        Method to delete vrf's by their id's

        :param ids: Identifiers of vrf's
        :return: None
        """
        url = build_uri_with_ids('api/v3/vrf/%s/', ids)

        return super(ApiVrf, self).delete(url)