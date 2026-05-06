def start(self):
        """
        Starts listening to the socket

        :return: True if the socket has been created
        """
        # Create the multicast socket (update the group)
        self._socket, self._group = create_multicast_socket(self._group,
                                                            self._port)

        # Start the listening thread
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self.__read,
            name="MulticastReceiver-{0}".format(self._port))
        self._thread.start()