def devices(self):
        """ computes the name of the disk devices that are suitable
        installation targets by subtracting CDROM- and USB devices
        from the list of total mounts.
        """
        install_devices = self.install_devices
        if 'bootstrap-system-devices' in env.instance.config:
            devices = set(env.instance.config['bootstrap-system-devices'].split())
        else:
            devices = set(self.sysctl_devices)
            for sysctl_device in self.sysctl_devices:
                for install_device in install_devices:
                    if install_device.startswith(sysctl_device):
                        devices.remove(sysctl_device)
        return devices