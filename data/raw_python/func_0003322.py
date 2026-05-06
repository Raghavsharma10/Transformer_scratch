async def close(self):
        """
        Close this request, send all data. You can still run other operations in the handler.
        """
        if not self._sendHeaders:
            self._startResponse()
        if self.inputstream is not None:
            self.inputstream.close(self.connection.scheduler)
        if self.outputstream is not None:
            await self.flush(True)
        if hasattr(self, 'session') and self.session:
            self.session.unlock()