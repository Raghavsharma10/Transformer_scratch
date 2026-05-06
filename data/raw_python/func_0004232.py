def update_channel(self, channel):
        """
        Method to update a channel.
        :param channel: List containing channel's desired to be created on database.
        :return: Id.
        """

        data = {'channels': channel}
        return super(ApiInterfaceRequest, self).put('api/v3/channel/', data)