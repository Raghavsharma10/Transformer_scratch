def update(self, ipv4s):
        """
        Method to update ipv4's

        :param ipv4s: List containing ipv4's desired to updated
        :return: None
        """

        data = {'ips': ipv4s}
        ipv4s_ids = [str(ipv4.get('id')) for ipv4 in ipv4s]

        return super(ApiIPv4, self).put('api/v3/ipv4/%s/' %
                                        ';'.join(ipv4s_ids), data)