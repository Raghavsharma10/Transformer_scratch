def sendall_stderr(self, s):
        """
        Send data to the channel's "stderr" stream, without allowing partial
        results.  Unlike L{send_stderr}, this method continues to send data
        from the given string until all data has been sent or an error occurs.
        Nothing is returned.
        
        @param s: data to send to the client as "stderr" output.
        @type s: str
        
        @raise socket.timeout: if sending stalled for longer than the timeout
            set by L{settimeout}.
        @raise socket.error: if an error occured before the entire string was
            sent.
            
        @since: 1.1
        """
        while s:
            if self.closed:
                raise socket.error('Socket is closed')
            sent = self.send_stderr(s)
            s = s[sent:]
        return None