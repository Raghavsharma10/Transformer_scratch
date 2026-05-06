def undeploy(self, ids, clean_up=0):
        """
        Method to undeploy vip's

        :param vips: List containing vip's desired to be undeployed on equipment
        :return: None
        """
        url = build_uri_with_ids('api/v3/vip-request/deploy/%s/?cleanup=%s', ids, clean_up)

        return super(ApiVipRequest, self).delete(url)