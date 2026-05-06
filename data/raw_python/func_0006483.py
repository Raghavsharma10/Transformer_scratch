def _read_header(self):
        '''Read and parse the header and calculate derived keys.'''

        try:
            # Read the entire header
            header = self._fp.read(96)
            if len(header) != 96:
                raise InvalidScryptFileFormat("Incomplete header")

            # Magic number
            if header[0:6] != b'scrypt':
                raise InvalidScryptFileFormat('Invalid magic number").')

            # Version (we only support 0)
            version = get_byte(header[6])
            if version != 0:
                raise InvalidScryptFileFormat('Unsupported version (%d)' % version)

            # Scrypt parameters
            self._N = 1 << get_byte(header[7])
            (self._r, self._p) = struct.unpack('>II', header[8:16])
            self._salt = header[16:48]

            # Generate the key
            self._key = hash(self._password, self._salt, self._N, self._r, self._p, 64)

            # Header Checksum
            checksum = header[48:64]
            calculate_checksum = hashlib.sha256(header[0:48]).digest()[:16]
            if checksum != calculate_checksum:
                raise InvalidScryptFileFormat('Incorrect header checksum')

            # Stream checksum
            checksum = header[64:96]
            self._checksumer = hmac.new(self.key[32:], msg = header[0:64], digestmod = hashlib.sha256)
            if checksum != self._checksumer.digest():
                raise InvalidScryptFileFormat('Incorrect header stream checksum')
            self._checksumer.update(header[64:96])

            # Prepare the AES engine
            self._crypto = aesctr.AESCounterModeOfOperation(key = self.key[:32])

            self._done_header = True

        except InvalidScryptFileFormat as e:
            self.close()
            raise e

        except Exception as e:
            self.close()
            raise InvalidScryptFileFormat('Header error (%s)' % e)