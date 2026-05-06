def notify(self):
        """
        Calls the notification method

        :return: True if the notification method has been called
        """
        if self.__method is not None:
            self.__method(self.__peer)
            return True
        return False