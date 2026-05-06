def write_name(self, name):
        """Writes a domain name to the packet"""

        try:
            # Find existing instance of this name in packet
            #
            index = self.names[name]
        except KeyError:
            # No record of this name already, so write it
            # out as normal, recording the location of the name
            # for future pointers to it.
            #
            self.names[name] = self.size
            parts = name.split('.')
            if parts[-1] == '':
                parts = parts[:-1]
            for part in parts:
                self.write_utf(part)
            self.write_byte(0)
            return

        # An index was found, so write a pointer to it
        #
        self.write_byte((index >> 8) | 0xC0)
        self.write_byte(index)