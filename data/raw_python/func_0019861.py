def _sendAction(self, action, attrs=None, chan_vars=None):
        """Send action to Asterisk Manager Interface.
        
        @param action:    Action name
        @param attrs:     Tuple of key-value pairs for action attributes.
        @param chan_vars: Tuple of key-value pairs for channel variables.

        """
        self._conn.write("Action: %s\r\n" % action)
        if attrs:
            for (key,val) in attrs:
                self._conn.write("%s: %s\r\n" % (key, val))
        if chan_vars:
            for (key,val) in chan_vars:
                self._conn.write("Variable: %s=%s\r\n" % (key, val))
        self._conn.write("\r\n")