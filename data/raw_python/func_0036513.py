def decrypt(self, cipherText):
    """Decrypt an arbitrary-length block of data.

    NOTE: This function formerly worked only on 16-byte blocks of `cipherText`.
    code that assumed this should still work fine, but can optionally be
    modified to call `decrypt_block` instead.

    Args:
        cipherText (str): data to decrypt. If the data is not a multiple of 16
            bytes long, it will be padded with null (0x00) bytes until it is.
            WARNING: This is almost certainty never need to happen for
            correctly-encrypted data.

    Returns:
        decrypted data. Note that this will always be a multiple of 16 bytes
            long. If the original data was not a multiple of 16 bytes, the
            result will contain trailing null bytes, which can be removed with
            `.rstrip('\x00')`
    """
    decryptedResult = ''
    for index in range(0, len(cipherText), BLOCK_SIZE):
      block = cipherText[index:index + BLOCK_SIZE]
      # Pad to required length if needed
      if len(block) < BLOCK_SIZE:
        block = zero_pad(block, BLOCK_SIZE)
      decryptedResult += self.decrypt_block(block)
    return decryptedResult