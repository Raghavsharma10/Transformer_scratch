def _write_header(self):
        'Writes the header to the underlying file object.'

        header = b'scrypt' + CHR0 + struct.pack('>BII', int(math.log(self.N, 2)), self.r, self.p) + self.salt

        # Add the header checksum to the header
        checksum = hashlib.sha256(header).digest()[:16]
        header += checksum

        # Add the header stream checksum
        self._checksumer = hmac.new(self.key[32:], msg = header, digestmod = hashlib.sha256)
        checksum = self._checksumer.digest()
        header += checksum
        self._checksumer.update(checksum)

        # Write the header
        self._fp.write(header)

        # Prepare the AES engine
        self._crypto = aesctr.AESCounterModeOfOperation(key = self.key[:32])
        #self._crypto = aes(self.key[:32])

        self._done_header = True