def open(self):
        """Open a connection to the device."""
        device_type = 'cisco_ios'
        if self.transport == 'telnet':
            device_type = 'cisco_ios_telnet'
        self.device = ConnectHandler(device_type=device_type,
                                     host=self.hostname,
                                     username=self.username,
                                     password=self.password,
                                     **self.netmiko_optional_args)
        # ensure in enable mode
        self.device.enable()