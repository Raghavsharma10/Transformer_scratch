def decrypt_block(self, cipherText):
    """Decrypt a 16-byte block of data.

    NOTE: This function was formerly called `decrypt`, but was changed when
    support for decrypting arbitrary-length strings was added.

    Args:
        cipherText (str): 16-byte data.

    Returns:
        16-byte str.

    Raises:
        TypeError if CamCrypt object has not been initialized.
        ValueError if `cipherText` is not BLOCK_SIZE (i.e. 16) bytes.
    """
    if not self.initialized:
      raise TypeError("CamCrypt object has not been initialized")
    if len(cipherText) != BLOCK_SIZE:
      raise ValueError("cipherText must be %d bytes long (received %d bytes)" %
                       (BLOCK_SIZE, len(cipherText)))
    plain = ctypes.create_string_buffer(BLOCK_SIZE)
    self.decblock(self.bitlen, cipherText, self.keytable, plain)
    return plain.raw