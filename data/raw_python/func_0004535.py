def list(self, environment_vip=None):
        """List networks redeipv6 ]

        :param environment_vip: environment vip to filter

        :return: IPv6 Networks
        """

        uri = 'api/networkv6/?'
        if environment_vip:
            uri += 'environment_vip=%s' % environment_vip

        return super(ApiNetworkIPv6, self).get(uri)