def decode(self, check_trailer=False): # pylint: disable=I0011,R0912
        """ Decode data in C{self.data} and return deserialized object.

            @param check_trailer: Raise error if trailing junk is found in data?
            @raise BencodeError: Invalid data.
        """
        try:
            kind = self.data[self.offset]
        except IndexError:
            raise BencodeError("Unexpected end of data at offset %d/%d" % (
                self.offset, len(self.data),
            ))

        if kind.isdigit():
            # String
            try:
                end = self.data.find(':', self.offset)
                length = int(self.data[self.offset:end], 10)
            except (ValueError, TypeError):
                raise BencodeError("Bad string length at offset %d (%r...)" % (
                    self.offset, self.data[self.offset:self.offset+32]
                ))

            self.offset = end+length+1
            obj = self.data[end+1:self.offset]

            if self.char_encoding:
                try:
                    obj = obj.decode(self.char_encoding)
                except (UnicodeError, AttributeError):
                    # deliver non-decodable string (byte arrays) as-is
                    pass
        elif kind == 'i':
            # Integer
            try:
                end = self.data.find('e', self.offset+1)
                obj = int(self.data[self.offset+1:end], 10)
            except (ValueError, TypeError):
                raise BencodeError("Bad integer at offset %d (%r...)" % (
                    self.offset, self.data[self.offset:self.offset+32]
                ))
            self.offset = end+1
        elif kind == 'l':
            # List
            self.offset += 1
            obj = []
            while self.data[self.offset:self.offset+1] != 'e':
                obj.append(self.decode())
            self.offset += 1
        elif kind == 'd':
            # Dict
            self.offset += 1
            obj = {}
            while self.data[self.offset:self.offset+1] != 'e':
                key = self.decode()
                obj[key] = self.decode()
            self.offset += 1
        else:
            raise BencodeError("Format error at offset %d (%r...)" % (
                self.offset, self.data[self.offset:self.offset+32]
            ))

        if check_trailer and self.offset != len(self.data):
            raise BencodeError("Trailing data at offset %d (%r...)" % (
                self.offset, self.data[self.offset:self.offset+32]
            ))

        return obj