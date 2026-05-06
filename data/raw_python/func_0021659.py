def _send_merge_commands(self, config, file_config):
        """
        Netmiko is being used to push set commands.
        """
        if self.loaded is False:
            if self._save_backup() is False:
                raise MergeConfigException('Error while storing backup '
                                           'config.')
        if self.ssh_connection is False:
            self._open_ssh()

        if file_config:
            if isinstance(config, str):
                config = config.splitlines()
        else:
            if isinstance(config, str):
                config = str(config).split()

        self.ssh_device.send_config_set(config)
        self.loaded = True
        self.merge_config = True