def check_vip_ip(self, ip, environment_vip):
        """
        Check available ip in environment vip
        """
        uri = 'api/ipv4/ip/%s/environment-vip/%s/' % (ip, environment_vip)

        return super(ApiNetworkIPv4, self).get(uri)