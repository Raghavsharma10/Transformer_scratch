def _ip_is_usable(self, current_ip):
        """
        Check if the current Tor's IP is usable.

        :argument current_ip: current Tor IP
        :type current_ip: str

        :returns bool
        """
        # Consider IP addresses only.
        try:
            ipaddress.ip_address(current_ip)
        except ValueError:
            return False

        # Never use real IP.
        if current_ip == self.real_ip:
            return False

        # Do dot allow IP reuse.
        if not self._ip_is_safe(current_ip):
            return False

        return True