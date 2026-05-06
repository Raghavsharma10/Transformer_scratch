def get_ip(self):
        """
        Retrieve a complete list of bought ip address related only to PRO Servers.
        It create an internal object (Iplist) representing all of the ips object
        iterated form the WS.
        @param: None
        @return: None
        """
        json_scheme = self.gen_def_json_scheme('GetPurchasedIpAddresses')
        json_obj = self.call_method_post(method='GetPurchasedIpAddresses ', json_scheme=json_scheme)
        self.iplist = IpList()
        for ip in json_obj['Value']:
            r = Ip()
            r.ip_addr = ip['Value']
            r.resid = ip['ResourceId']
            r.serverid = ip['ServerId'] if 'None' not in str(ip['ServerId']) else None
            self.iplist.append(r)