def encrypt(self, plainText):
    """Encrypt an arbitrary-length block of data.

    NOTE: This function formerly worked only on 16-byte blocks of `plainText`.
    code that assumed this should still work fine, but can optionally be
    modified to call `encrypt_block` instead.

    Args:
        plainText (str): data to encrypt. If the data is not a multiple of 16
            bytes long, it will be padded with null (0x00) bytes until it is.

    Returns:
        encrypted data. Note that this will always be a multiple of 16 bytes
            long.
    """
    encryptedResult = ''
    for index in range(0, len(plainText), BLOCK_SIZE):
      block = plainText[index:index + BLOCK_SIZE]
      # Pad to required length if needed
      if len(block) < BLOCK_SIZE:
        block = zero_pad(block, BLOCK_SIZE)
      encryptedResult += self.encrypt_block(block)
    return encryptedResult