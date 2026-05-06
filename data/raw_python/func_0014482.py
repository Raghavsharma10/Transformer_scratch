def add_int(self, n):
        """
        Add an integer to the stream.
        
        @param n: integer to add
        @type n: int
        """
        self.packet.write(struct.pack('>I', n))
        return self