def save_vip_request(self, vip_request):
        """
        Method to save vip request

        param vip_request: vip_request object
        """
        uri = 'api/v3/vip-request/'

        data = dict()
        data['vips'] = list()
        data['vips'].append(vip_request)

        return super(ApiVipRequest, self).post(uri, data)