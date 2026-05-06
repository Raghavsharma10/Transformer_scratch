def _decrypt(file_contents, password):
  """
  The corresponding decryption routine for _encrypt().

  'securesystemslib.exceptions.CryptoError' raised if the decryption fails.
  """

  # Extract the salt, iterations, hmac, initialization vector, and ciphertext
  # from 'file_contents'.  These five values are delimited by
  # '_ENCRYPTION_DELIMITER'.  This delimiter is arbitrarily chosen and should
  # not occur in the hexadecimal representations of the fields it is
  # separating.  Raise 'securesystemslib.exceptions.CryptoError', if
  # 'file_contents' does not contains the expected data layout.
  try:
    salt, iterations, hmac, iv, ciphertext = \
      file_contents.split(_ENCRYPTION_DELIMITER)

  except ValueError:
    raise securesystemslib.exceptions.CryptoError('Invalid encrypted file.')

  # Ensure we have the expected raw data for the delimited cryptographic data.
  salt = binascii.unhexlify(salt.encode('utf-8'))
  iterations = int(iterations)
  iv = binascii.unhexlify(iv.encode('utf-8'))
  ciphertext = binascii.unhexlify(ciphertext.encode('utf-8'))

  # Generate derived key from 'password'.  The salt and iterations are
  # specified so that the expected derived key is regenerated correctly.
  # Discard the old "salt" and "iterations" values, as we only need the old
  # derived key.
  junk_old_salt, junk_old_iterations, symmetric_key = \
    _generate_derived_key(password, salt, iterations)

  # Verify the hmac to ensure the ciphertext is valid and has not been altered.
  # See the encryption routine for why we use the encrypt-then-MAC approach.
  # The decryption routine may verify a ciphertext without having to perform
  # a decryption operation.
  generated_hmac_object = \
    cryptography.hazmat.primitives.hmac.HMAC(symmetric_key, hashes.SHA256(),
        backend=default_backend())
  generated_hmac_object.update(ciphertext)
  generated_hmac = binascii.hexlify(generated_hmac_object.finalize())


  if not securesystemslib.util.digests_are_equal(generated_hmac.decode(), hmac):
    raise securesystemslib.exceptions.CryptoError('Decryption failed.')

  # Construct a Cipher object, with the key and iv.
  decryptor = Cipher(algorithms.AES(symmetric_key), modes.CTR(iv),
      backend=default_backend()).decryptor()

  # Decryption gets us the authenticated plaintext.
  plaintext = decryptor.update(ciphertext) + decryptor.finalize()

  return plaintext