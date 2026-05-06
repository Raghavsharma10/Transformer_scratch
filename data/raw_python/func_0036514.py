def encrypt_block(self, plainText):
    """Encrypt a 16-byte block of data.

    NOTE: This function was formerly called `encrypt`, but was changed when
    support for encrypting arbitrary-length strings was added.

    Args:
        plainText (str): 16-byte data.

    Returns:
        16-byte str.

    Raises:
        TypeError if CamCrypt object has not been initialized.
        ValueError if `plainText` is not BLOCK_SIZE (i.e. 16) bytes.
    """
    if not self.initialized:
      raise TypeError("CamCrypt object has not been initialized")
    if len(plainText) != BLOCK_SIZE:
      raise ValueError("plainText must be %d bytes long (received %d bytes)" %
                       (BLOCK_SIZE, len(plainText)))
    cipher = ctypes.create_string_buffer(BLOCK_SIZE)
    self.encblock(self.bitlen, plainText, self.keytable, cipher)
    return cipher.raw