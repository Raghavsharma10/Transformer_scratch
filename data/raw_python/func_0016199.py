def reinitialize(self, admin_password=None, debug=False, ConfigureIPv6=False, OSTemplateID=None):
        """
        Reinitialize a VM.
        :param admin_password: Administrator password.
        :param debug: Flag to enable debug output.
        :param ConfigureIPv6: Flag to enable IPv6 on the VM.
        :param OSTemplateID: TemplateID to reinitialize the VM with.
        :return: True in case of success, otherwise False
        :type admin_password: str
        :type debug: bool
        :type ConfigureIPv6: bool
        :type OSTemplateID: int
        """
        data = dict(
            AdministratorPassword=admin_password,
            ServerId=self.sid,
            ConfigureIPv6=ConfigureIPv6
        )
        if OSTemplateID is not None:
            data.update(OSTemplateID=OSTemplateID)
        assert data['AdministratorPassword'] is not None, 'Error reinitializing VM: no admin password specified.'
        assert data['ServerId'] is not None, 'Error reinitializing VM: no Server Id specified.'
        json_scheme = self.interface.gen_def_json_scheme('SetEnqueueReinitializeServer', method_fields=data)
        json_obj = self.interface.call_method_post('SetEnqueueReinitializeServer', json_scheme=json_scheme, debug=debug)
        return True if json_obj['Success'] is 'True' else False