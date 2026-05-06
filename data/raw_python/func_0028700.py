def _encrypt(key_data, derived_key_information):
  """
  Encrypt 'key_data' using the Advanced Encryption Standard (AES-256) algorithm.
  'derived_key_information' should contain a key strengthened by PBKDF2.  The
  key size is 256 bits and AES's mode of operation is set to CTR (CounTeR Mode).
  The HMAC of the ciphertext is generated to ensure the ciphertext has not been
  modified.

  'key_data' is the JSON string representation of the key.  In the case
  of RSA keys, this format would be 'securesystemslib.formats.RSAKEY_SCHEMA':

  {'keytype': 'rsa',
   'keyval': {'public': '-----BEGIN RSA PUBLIC KEY----- ...',
              'private': '-----BEGIN RSA PRIVATE KEY----- ...'}}

  'derived_key_information' is a dictionary of the form:
    {'salt': '...',
     'derived_key': '...',
     'iterations': '...'}

  'securesystemslib.exceptions.CryptoError' raised if the encryption fails.
  """

  # Generate a random Initialization Vector (IV).  Follow the provably secure
  # encrypt-then-MAC approach, which affords the ability to verify ciphertext
  # without needing to decrypt it and preventing an attacker from feeding the
  # block cipher malicious data.  Modes like GCM provide both encryption and
  # authentication, whereas CTR only provides encryption.

  # Generate a random 128-bit IV.  Random bits of data is needed for salts and
  # initialization vectors suitable for the encryption algorithms used in
  # 'pyca_crypto_keys.py'.
  iv = os.urandom(16)

  # Construct an AES-CTR Cipher object with the given key and a randomly
  # generated IV.
  symmetric_key = derived_key_information['derived_key']
  encryptor = Cipher(algorithms.AES(symmetric_key), modes.CTR(iv),
      backend=default_backend()).encryptor()

  # Encrypt the plaintext and get the associated ciphertext.
  # Do we need to check for any exceptions?
  ciphertext = encryptor.update(key_data.encode('utf-8')) + encryptor.finalize()

  # Generate the hmac of the ciphertext to ensure it has not been modified.
  # The decryption routine may verify a ciphertext without having to perform
  # a decryption operation.
  symmetric_key = derived_key_information['derived_key']
  salt = derived_key_information['salt']
  hmac_object = \
    cryptography.hazmat.primitives.hmac.HMAC(symmetric_key, hashes.SHA256(),
        backend=default_backend())
  hmac_object.update(ciphertext)
  hmac_value = binascii.hexlify(hmac_object.finalize())

  # Store the number of PBKDF2 iterations used to derive the symmetric key so
  # that the decryption routine can regenerate the symmetric key successfully.
  # The PBKDF2 iterations are allowed to vary for the keys loaded and saved.
  iterations = derived_key_information['iterations']

  # Return the salt, iterations, hmac, initialization vector, and ciphertext
  # as a single string.  These five values are delimited by
  # '_ENCRYPTION_DELIMITER' to make extraction easier.  This delimiter is
  # arbitrarily chosen and should not occur in the hexadecimal representations
  # of the fields it is separating.
  return binascii.hexlify(salt).decode() + _ENCRYPTION_DELIMITER + \
      str(iterations) + _ENCRYPTION_DELIMITER + \
      hmac_value.decode() + _ENCRYPTION_DELIMITER + \
      binascii.hexlify(iv).decode() + _ENCRYPTION_DELIMITER + \
      binascii.hexlify(ciphertext).decode()