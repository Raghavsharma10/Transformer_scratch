async def write(self, data, eof = False, buffering = True):
        """
        Write output to current output stream
        """
        if not self.outputstream:
            self.outputstream = Stream()
            self._startResponse()
        elif (not buffering or eof) and not self._sendHeaders:
            self._startResponse()
        if not isinstance(data, bytes):
            data = data.encode(self.encoding)
        await self.outputstream.write(data, self.connection, eof, False, buffering)