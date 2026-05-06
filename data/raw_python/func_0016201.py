def poweroff_server(self, server=None, server_id=None):
        """
        Poweroff a VM. If possible to pass the VM object or simply the ID
        of the VM that we want to turn on.
        Args:
            server: VM Object that represent the VM to power off,
            server_id: Int or Str representing the ID of the VM to power off.
        Returns:
            return True if json_obj['Success'] is 'True' else False
        """
        sid = server_id if server_id is not None else server.sid
        if sid is None:
            raise Exception('No Server Specified.')
        json_scheme = self.gen_def_json_scheme('SetEnqueueServerPowerOff', dict(ServerId=sid))
        json_obj = self.call_method_post('SetEnqueueServerPowerOff', json_scheme=json_scheme)
        return True if json_obj['Success'] is 'True' else False