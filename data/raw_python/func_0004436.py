def deploy(self, ids):
        """
        Method to deploy vip's

        :param vips: List containing vip's desired to be deployed on equipment
        :return: None
        """
        url = build_uri_with_ids('api/v3/vip-request/deploy/%s/', ids)

        return super(ApiVipRequest, self).post(url)