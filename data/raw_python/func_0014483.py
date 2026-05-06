def add_string(self, s):
        """
        Add a string to the stream.
        
        @param s: string to add
        @type s: str
        """
        self.add_int(len(s))
        self.packet.write(s)
        return self