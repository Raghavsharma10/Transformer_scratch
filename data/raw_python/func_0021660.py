def compare_config(self):
        """
        Netmiko is being used to obtain config diffs because pan-python
        doesn't support the needed command.
        """
        if self.ssh_connection is False:
            self._open_ssh()

        self.ssh_device.exit_config_mode()
        diff = self.ssh_device.send_command("show config diff")
        return diff.strip()