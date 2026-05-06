def set_send_enable(self, setting):
        """
        Set the send enable setting on the watch
        """
        self._pebble.send_packet(DataLogging(data=DataLoggingSetSendEnable(enabled=setting)))