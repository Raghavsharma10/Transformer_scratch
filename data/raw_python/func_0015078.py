def _unescape(self, msg):
        """
        Removes double quotes that were used to escape double quotes. Expects
        a string without its delimiting quotes, or a number. Returns a new
        unescaped string.
        """
        if isinstance(msg, (int, float, long)):
            return msg

        unescaped = ''
        i = 0
        while i < len(msg):
            unescaped += msg[i]
            if msg[i] == '"':
                i+=1
            i+=1
        return unescaped