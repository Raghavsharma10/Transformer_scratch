def get_vip_request_details(self, vip_request_id):
        """
        Method to get details of vip request

        param vip_request_id: vip_request id
        """
        uri = 'api/v3/vip-request/details/%s/' % vip_request_id

        return super(ApiVipRequest, self).get(uri)