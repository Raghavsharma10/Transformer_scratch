def add_channels(self, channels):
        """
        Take a list of SockChannel objects and extend the websock listener
        """
        chans = [
            SockChannel(chan, res, self._generate_result)
            for chan, res in channels.items()
        ]
        self.api.channels.extend(chans)
        self.api.connect_channels(chans)