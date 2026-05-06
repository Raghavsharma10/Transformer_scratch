def create(self, vips):
        """
        Method to create vip's

        :param vips: List containing vip's desired to be created on database
        :return: None
        """

        data = {'vips': vips}
        return super(ApiVipRequest, self).post('api/v3/vip-request/', data)