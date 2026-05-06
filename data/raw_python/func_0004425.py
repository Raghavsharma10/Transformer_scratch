def get_vip_request(self, vip_request_id):
        """
        Method to get vip request

        param vip_request_id: vip_request id
        """
        uri = 'api/v3/vip-request/%s/' % vip_request_id

        return super(ApiVipRequest, self).get(uri)