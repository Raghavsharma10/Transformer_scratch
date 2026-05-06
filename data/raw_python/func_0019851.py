def retrieveVals(self):
        """Retrieve values for graphs."""
        fs = FSinfo(self._fshost, self._fsport, self._fspass)
        if self.hasGraph('fs_calls'):
            count = fs.getCallCount()
            self.setGraphVal('fs_calls', 'calls', count)
        if self.hasGraph('fs_channels'):
            count = fs.getChannelCount()
            self.setGraphVal('fs_channels', 'channels', count)