def _get_keyid(keytype, scheme, key_value, hash_algorithm = 'sha256'):
  """Return the keyid of 'key_value'."""

  # 'keyid' will be generated from an object conformant to KEY_SCHEMA,
  # which is the format Metadata files (e.g., root.json) store keys.
  # 'format_keyval_to_metadata()' returns the object needed by _get_keyid().
  key_meta = format_keyval_to_metadata(keytype, scheme, key_value, private=False)

  # Convert the key to JSON Canonical format, suitable for adding
  # to digest objects.
  key_update_data = securesystemslib.formats.encode_canonical(key_meta)

  # Create a digest object and call update(), using the JSON
  # canonical format of 'rskey_meta' as the update data.
  digest_object = securesystemslib.hash.digest(hash_algorithm)
  digest_object.update(key_update_data.encode('utf-8'))

  # 'keyid' becomes the hexadecimal representation of the hash.
  keyid = digest_object.hexdigest()

  return keyid