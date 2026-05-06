def connect(self, host, port=6667, password=None):
        """
        Connects to a server
        """
        # Prepare the callbacks
        self._irc.add_global_handler('all_events', self.__handler)

        # Prepare the connection
        self._connection = self._irc.server().connect(
            host, port, self._nickname, password,
            self._username, self._realname)

        # Start connection thread
        self.__stopped.clear()
        self.__thread = threading.Thread(target=self.__loop,
                                         name="IRC-Bot-Loop")
        self.__thread.daemon = True
        self.__thread.start()