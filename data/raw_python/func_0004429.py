def delete_vip_request(self, vip_request_ids):
        """
        Method to delete vip request

        param vip_request_ids: vip_request ids
        """
        uri = 'api/v3/vip-request/%s/' % vip_request_ids

        return super(ApiVipRequest, self).delete(uri)