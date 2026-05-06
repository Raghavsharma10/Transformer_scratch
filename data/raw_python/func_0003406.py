async def write(self, data, container, eof = False, ignoreexception = False, buffering = True, split = True):
        """
        Coroutine method to write data to this stream.
        
        :param data: data to write
        
        :param container: the routine container
        
        :param eof: if True, this is the last chunk of this stream. The other end will receive an EOF after reading
                    this chunk.
                    
        :param ignoreexception: even if the stream is closed on the other side, do not raise exception.
        
        :param buffering: enable buffering. The written data may not be sent if buffering = True; if buffering = False,
                          immediately send any data in the buffer together with this chunk.
        
        :param split: enable splitting. If this chunk is too large, the stream is allowed to split it into
                      smaller chunks for better balancing.
        """
        if not ignoreexception:
            raise IOError('Stream is closed')