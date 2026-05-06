def search_vip_request(self, search):
        """
        Method to list vip request

        param search: search
        """
        uri = 'api/v3/vip-request/?%s' % urllib.urlencode({'search': search})

        return super(ApiVipRequest, self).get(uri)