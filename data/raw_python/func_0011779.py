async def _check_ip(self, addr):
        """
        Async check ip with dnsbl providers.
        Parameters:
            * addr - ip address to check

        Returns:
            * DNSBLResult object
        """

        tasks = []
        for provider in self.providers:
            tasks.append(self.dnsbl_request(addr, provider))
        results = await asyncio.gather(*tasks)
        return DNSBLResult(addr=addr, results=results)