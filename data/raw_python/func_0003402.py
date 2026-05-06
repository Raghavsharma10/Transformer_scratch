async def read(self, container = None, size = None):
        """
        Coroutine method to read from the stream and return the data. Raises EOFError
        when the stream end has been reached; raises IOError if there are other errors.
        
        :param container: A routine container
        
        :param size: maximum read size, or unlimited if is None
        """
        ret = []
        retsize = 0
        if self.eof:
            raise EOFError
        if self.errored:
            raise IOError('Stream is broken before EOF')
        while size is None or retsize < size:
            if self.pos >= len(self.data):
                await self.prepareRead(container)
            if size is None or size - retsize >= len(self.data) - self.pos:
                t = self.data[self.pos:]
                ret.append(t)
                retsize += len(t)
                self.pos = len(self.data)
                if self.dataeof:
                    self.eof = True
                    break
                if self.dataerror:
                    self.errored = True
                    break
            else:
                ret.append(self.data[self.pos:self.pos + (size - retsize)])
                self.pos += (size - retsize)
                retsize = size
                break
        if self.isunicode:
            return u''.join(ret)
        else:
            return b''.join(ret)
        if self.errored:
            raise IOError('Stream is broken before EOF')