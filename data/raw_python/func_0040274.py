def inline(self) -> str:
        """
        Return endpoint string

        :return:
        """
        return BMAEndpoint.API + "{DNS}{IPv4}{IPv6}{PORT}" \
            .format(DNS=(" {0}".format(self.server) if self.server else ""),
                    IPv4=(" {0}".format(self.ipv4) if self.ipv4 else ""),
                    IPv6=(" {0}".format(self.ipv6) if self.ipv6 else ""),
                    PORT=(" {0}".format(self.port) if self.port else ""))