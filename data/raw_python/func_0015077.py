def _escape(self, msg):
        """
        Escapes double quotes by adding another double quote as per the Scratch
        protocol. Expects a string without its delimiting quotes. Returns a new
        escaped string. 
        """
        escaped = ''	
        for c in msg: 
            escaped += c
            if c == '"':
                escaped += '"'
        return escaped