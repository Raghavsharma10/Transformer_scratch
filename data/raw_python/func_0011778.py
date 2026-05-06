async def dnsbl_request(self, addr, provider):
        """
        Make lookup to dnsbl provider
        Parameters:
            * addr (string) - ip address to check
            * provider (string) - dnsbl provider

        Returns:
            * DNSBLResponse object

        Raises:
            * ValueError
        """
        response = None
        error = None
        try:
            socket.inet_aton(addr)
        except socket.error:
            raise ValueError('wrong ip format')
        ip_reversed = '.'.join(reversed(addr.split('.')))
        dnsbl_query = "%s.%s" % (ip_reversed, provider.host)
        try:
            async with self._semaphore:
                response = await self._resolver.query(dnsbl_query, 'A')
        except aiodns.error.DNSError as exc:
            if exc.args[0] != 4: # 4: domain name not found:
                error = exc

        return DNSBLResponse(addr=addr, provider=provider, response=response, error=error)