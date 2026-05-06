def check_vip_ip(self, ip, environment_vip):
        """
        Check available ipv6 in environment vip
        """
        uri = 'api/ipv6/ip/%s/environment-vip/%s/' % (ip, environment_vip)

        return super(ApiNetworkIPv6, self).get(uri)