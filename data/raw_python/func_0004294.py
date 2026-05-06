def update(self, ipv6s):
        """
        Method to update ipv6's

        :param ipv6s: List containing ipv6's desired to updated
        :return: None
        """

        data = {'ips': ipv6s}
        ipv6s_ids = [str(ipv6.get('id')) for ipv6 in ipv6s]

        return super(ApiIPv6, self).put('api/v3/ipv6/%s/' %
                                        ';'.join(ipv6s_ids), data)