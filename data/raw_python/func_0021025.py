def getOneMessage ( self ):
        """
        I pull one complete message off the buffer and return it decoded
        as a dict. If there is no complete message in the buffer, I
        return None.

        Note that the buffer can contain more than once message. You
        should therefore call me in a loop until I return None.
        """
        ( mbytes, hbytes ) = self._findMessageBytes ( self.buffer )
        if not mbytes:
            return None
        
        msgdata = self.buffer[:mbytes]
        self.buffer = self.buffer[mbytes:]
        hdata = msgdata[:hbytes]
        elems = hdata.split ( '\n' )
        cmd     = elems.pop ( 0 )
        headers = {}
        # We can't use a simple split because the value can legally contain
        # colon characters (for example, the session returned by ActiveMQ).
        for e in elems:
            try:
                i = e.find ( ':' )
            except ValueError:
                continue
            k = e[:i].strip()
            v = e[i+1:].strip()
            headers [ k ] = v

        # hbytes points to the start of the '\n\n' at the end of the header,
        # so 2 bytes beyond this is the start of the body. The body EXCLUDES
        # the final two bytes, which are '\x00\n'. Note that these 2 bytes
        # are UNRELATED to the 2-byte '\n\n' that Frame.pack() used to insert
        # into the data stream.
        body = msgdata[hbytes+2:-2]
        msg = { 'cmd'     : cmd,
                'headers' : headers,
                'body'    : body,
                }
        return msg