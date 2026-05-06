def keygen(self, keyBitLength, rawKey):
    """ This must be called on the object before any encryption or
    decryption can take place.  Provide it the key bit length,
    which must be 128, 192, or 256, and the key, which may be a
    sequence of bytes or a simple string.
    Does not return any value.
    Raises an exception if the arguments are not sane.
    """
    if keyBitLength not in ACCEPTABLE_KEY_LENGTHS:
      raise Exception("keyBitLength must be 128, 192, or 256")
    self.bitlen = keyBitLength
    if len(rawKey) <= 0 or len(rawKey) > self.bitlen/8:
      raise Exception("rawKey must be less than or equal to keyBitLength/8 (%d) characters long" % (self.bitlen/8))
    rawKey = zero_pad(rawKey, self.bitlen/8)
    keytable = ctypes.create_string_buffer(TABLE_BYTE_LEN)
    self.ekeygen(self.bitlen, rawKey, keytable)
    self.keytable = keytable
    self.initialized = True