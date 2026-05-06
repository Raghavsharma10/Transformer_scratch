def __get_channel(self):
        "Get the channel to register webdriver to."
        if self.__config.get(WebDriverManager.ENABLE_THREADING_SUPPORT, False):
            channel = current_thread().ident
        else:
            channel = 0

        return channel