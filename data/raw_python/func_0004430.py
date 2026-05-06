def create_vip(self, vip_request_ids):
        """
        Method to create vip request

        param vip_request_ids: vip_request ids
        """
        uri = 'api/v3/vip-request/deploy/%s/' % vip_request_ids

        return super(ApiVipRequest, self).post(uri)