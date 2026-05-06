def option_vip_by_environment_vip_type(self, environment_vip_id, type_option):
        """
        List option vip.
        param environment_vip_id: Id of Environment Vip
        Param type_option: type option vip
        """

        uri = "api/v3/option-vip/environment-vip/%s/type-option/%s/" % (environment_vip_id, type_option)

        return self.get(uri)