def create_channel(self, channel):
        """
        Method to create a channel.
        :param channel: List containing channel's desired to be created on database.
        :return: Id.
        """

        data = {'channels': channel}
        return super(ApiInterfaceRequest, self).post('api/v3/channel/', data)