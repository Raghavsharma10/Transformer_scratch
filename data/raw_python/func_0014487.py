def sendall(self, s):
        """
        Send data to the channel, without allowing partial results.  Unlike
        L{send}, this method continues to send data from the given string until
        either all data has been sent or an error occurs.  Nothing is returned.

        @param s: data to send.
        @type s: str

        @raise socket.timeout: if sending stalled for longer than the timeout
            set by L{settimeout}.
        @raise socket.error: if an error occured before the entire string was
            sent.
        
        @note: If the channel is closed while only part of the data hase been
            sent, there is no way to determine how much data (if any) was sent.
            This is irritating, but identically follows python's API.
        """
        while s:
            if self.closed:
                # this doesn't seem useful, but it is the documented behavior of Socket
                raise socket.error('Socket is closed')
            sent = self.send(s)
            s = s[sent:]
        return None