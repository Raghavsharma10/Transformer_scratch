def _manage_used_ips(self, current_ip):
        """
        Handle registering and releasing used Tor IPs.

        :argument current_ip: current Tor IP
        :type current_ip: str
        """
        # Register current IP.
        self.used_ips.append(current_ip)

        # Release the oldest registred IP.
        if self.reuse_threshold:
            if len(self.used_ips) > self.reuse_threshold:
                del self.used_ips[0]