def purchase_ip(self, debug=False):
        """
        Return an ip object representing a new bought IP
        @param debug [Boolean] if true, request and response will be printed
        @return (Ip): Ip object
        """
        json_scheme = self.gen_def_json_scheme('SetPurchaseIpAddress')
        json_obj = self.call_method_post(method='SetPurchaseIpAddress', json_scheme=json_scheme, debug=debug)
        try:
            ip = Ip()
            ip.ip_addr = json_obj['Value']['Value']
            ip.resid = json_obj['Value']['ResourceId']
            return ip
        except:
            raise Exception('Unknown error retrieving IP.')