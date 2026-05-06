def register_token(self, *args, **kwargs):
        """
        Register token

        Accepts:
            - token_name [string]
            - contract_address [hex string]
            - blockchain [string]  token's blockchain (QTUMTEST, ETH)
        Returns dictionary with following fields:
            - success [Bool]
         """
        client = HTTPClient(self.withdraw_server_address + self.withdraw_endpoint)
        if check_sig:
            return client.request('register_token', self.signature_validator.sign(kwargs))
        else:
            return client.request('register_token', kwargs)