def is_valid_address(self, *args, **kwargs):
        """
        check address

        Accepts:
            - address [hex string] (withdrawal address in hex form)
            - coinid [string] (blockchain id (example: BTCTEST, LTCTEST))
        Returns dictionary with following fields:
            - bool [Bool]
        """

        client = HTTPClient(self.withdraw_server_address + self.withdraw_endpoint)

        return client.request('is_valid_address', kwargs)