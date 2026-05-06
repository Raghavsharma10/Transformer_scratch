def list(self, environment_vip=None):
        """List IPv4 networks

        :param environment_vip: environment vip to filter

        :return: IPv4 Networks
        """

        uri = 'api/networkv4/?'
        if environment_vip:
            uri += 'environment_vip=%s' % environment_vip

        return super(ApiNetworkIPv4, self).get(uri)