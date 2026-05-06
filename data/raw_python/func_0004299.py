def create(self, ipv4s):
        """
        Method to create ipv4's

        :param ipv4s: List containing ipv4's desired to be created on database
        :return: None
        """

        data = {'ips': ipv4s}
        return super(ApiIPv4, self).post('api/v3/ipv4/', data)