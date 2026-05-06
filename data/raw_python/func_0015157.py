def get_current_ip(self):
        """
        Get the current IP Tor is using.

        :returns str
        :raises TorIpError
        """
        response = get(ICANHAZIP, proxies={"http": self.local_http_proxy})

        if response.ok:
            return self._get_response_text(response)

        raise TorIpError("Failed to get the current Tor IP")