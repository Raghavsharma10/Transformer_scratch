def real_ip(self):
        """
        The actual public IP of this host.
        """
        if self._real_ip is None:
            response = get(ICANHAZIP)
            self._real_ip = self._get_response_text(response)

        return self._real_ip