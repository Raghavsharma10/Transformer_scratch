def broadcast(self, msg):
        """
        Broadcasts msg to Scratch. msg can be a single message or an iterable 
        (list, tuple, set, generator, etc.) of messages.
        """
        if getattr(msg, '__iter__', False): # iterable
            for m in msg:
                self._send('broadcast "%s"' % self._escape(str(m)))
        else: # probably a string or number
            self._send('broadcast "%s"' % self._escape(str(msg)))