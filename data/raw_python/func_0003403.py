def readonce(self, size = None):
        """
        Read from current buffer. If current buffer is empty, returns an empty string. You can use `prepareRead`
        to read the next chunk of data.
        
        This is not a coroutine method.
        """
        if self.eof:
            raise EOFError
        if self.errored:
            raise IOError('Stream is broken before EOF')
        if size is not None and size < len(self.data) - self.pos:
            ret = self.data[self.pos: self.pos + size]
            self.pos += size
            return ret
        else:
            ret = self.data[self.pos:]
            self.pos = len(self.data)
            if self.dataeof:
                self.eof = True
            return ret