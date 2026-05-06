def format(self, full_info: bool = False):
        """
        :param full_info: If True, adds more info about the chat. Please, note that this additional info requires
            to make up to THREE synchronous api calls.
        """
        chat = self.api_object
        if full_info:
            self.__format_full(chat)
        else:
            self.__format_simple(chat)