def __safe_errback(self, room_data, err_condition, err_text):
        """
        Safe use of the callback method, to avoid errors propagation

        :param room_data: A RoomData object
        :param err_condition: Category of error
        :param err_text: Description of the error
        """
        method = room_data.errback
        if method is not None:
            try:
                method(room_data.room, room_data.nick, err_condition, err_text)
            except Exception as ex:
                self.__logger.exception("Error calling back room creator: %s",
                                        ex)