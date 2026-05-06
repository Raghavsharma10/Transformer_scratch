async def readline(self, container = None, size = None):
        """
        Coroutine method which reads the next line or until EOF or size exceeds
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
                if self.isunicode:
                    p = t.find(u'\n')
                else:
                    p = t.find(b'\n')
                if p >= 0:
                    t = t[0: p + 1]
                    ret.append(t)
                    retsize += len(t)
                    self.pos += len(t)
                    break
                else:
                    ret.append(t)
                    retsize += len(t)
                    self.pos += len(t)
                    if self.dataeof:
                        self.eof = True
                        break
                    if self.dataerror:
                        self.errored = True
                        break
            else:
                t = self.data[self.pos:self.pos + (size - retsize)]
                if self.isunicode:
                    p = t.find(u'\n')
                else:
                    p = t.find(b'\n')
                if p >= 0:
                    t = t[0: p + 1]
                ret.append(t)
                self.pos += len(t)
                retsize += len(t)
                break
        if self.isunicode:
            return u''.join(ret)
        else:
            return b''.join(ret)
        if self.errored:
            raise IOError('Stream is broken before EOF')