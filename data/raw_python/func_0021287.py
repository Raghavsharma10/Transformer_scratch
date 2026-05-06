def dns_resolve(self):
        """ Perform DNS resolution on the contained addresses.

        :return:
        """
        new_addresses = []
        for address in self.addresses:
            try:
                info = getaddrinfo(address[0], address[1], 0, SOCK_STREAM, IPPROTO_TCP)
            except gaierror:
                raise AddressError("Cannot resolve address {!r}".format(address))
            else:
                for _, _, _, _, address in info:
                    if len(address) == 4 and address[3] != 0:
                        # skip any IPv6 addresses with a non-zero scope id
                        # as these appear to cause problems on some platforms
                        continue
                    new_addresses.append(address)
        self.addresses = new_addresses