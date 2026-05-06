def option_vip_by_environmentvip(self, environment_vip_id):
        """
        List Option Vip by Environment Vip

        param environment_vip_id: Id of Environment Vip
        """

        uri = 'api/v3/option-vip/environment-vip/%s/' % environment_vip_id

        return super(ApiVipRequest, self).get(uri)