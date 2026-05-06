def update(self, vips):
        """
        Method to update vip's

        :param vips: List containing vip's desired to updated
        :return: None
        """

        data = {'vips': vips}
        vips_ids = [str(vip.get('id')) for vip in vips]

        return super(ApiVipRequest, self).put('api/v3/vip-request/%s/' %
                                              ';'.join(vips_ids), data)