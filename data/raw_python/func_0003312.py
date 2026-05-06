def bufferoutput(self):
        """
        Buffer the whole output until write EOF or flushed.
        """
        new_stream = Stream(writebufferlimit=None)
        if self._sendHeaders:
            # An extra copy
            self.container.subroutine(new_stream.copy_to(self.outputstream, self.container, buffering=False))
        self.outputstream = Stream(writebufferlimit=None)