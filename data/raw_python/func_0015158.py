def get_new_ip(self):
        """
        Try to obtain new a usable TOR IP.

        :returns bool
        :raises TorIpError
        """
        attempts = 0

        while True:
            if attempts == self.new_ip_max_attempts:
                raise TorIpError("Failed to obtain a new usable Tor IP")

            attempts += 1

            try:
                current_ip = self.get_current_ip()
            except (RequestException, TorIpError):
                self._obtain_new_ip()
                continue

            if not self._ip_is_usable(current_ip):
                self._obtain_new_ip()
                continue

            self._manage_used_ips(current_ip)
            break

        return current_ip