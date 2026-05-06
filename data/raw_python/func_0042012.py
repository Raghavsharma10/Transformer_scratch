def __safe_callback(self, room_data):
        """
        Safe use of the callback method, to avoid errors propagation

        :param room_data: A RoomData object
        """
        method = room_data.callback
        if method is not None:
            try:
                method(room_data.room, room_data.nick)
            except Exception as ex:
                self.__logger.exception("Error calling back room creator: %s",
                                        ex)