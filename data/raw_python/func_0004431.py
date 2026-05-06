def remove_vip(self, vip_request_ids):
        """
        Method to delete vip request

        param vip_request_ids: vip_request ids
        """
        uri = 'api/v3/vip-request/deploy/%s/' % vip_request_ids

        return super(ApiVipRequest, self).delete(uri)