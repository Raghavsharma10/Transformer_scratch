def import_ed25519_publickey_from_file(filepath):
  """
  <Purpose>
    Load the ED25519 public key object (conformant to
    'securesystemslib.formats.KEY_SCHEMA') stored in 'filepath'.  Return
    'filepath' in securesystemslib.formats.ED25519KEY_SCHEMA format.

    If the key object in 'filepath' contains a private key, it is discarded.

  <Arguments>
    filepath:
      <filepath>.pub file, a public key file.

  <Exceptions>
    securesystemslib.exceptions.FormatError, if 'filepath' is improperly
    formatted or is an unexpected key type.

  <Side Effects>
    The contents of 'filepath' is read and saved.

  <Returns>
    An ED25519 key object conformant to
    'securesystemslib.formats.ED25519KEY_SCHEMA'.
  """

  # Does 'filepath' have the correct format?
  # Ensure the arguments have the appropriate number of objects and object
  # types, and that all dict keys are properly named.
  # Raise 'securesystemslib.exceptions.FormatError' if there is a mismatch.
  securesystemslib.formats.PATH_SCHEMA.check_match(filepath)

  # ED25519 key objects are saved in json and metadata format.  Return the
  # loaded key object in securesystemslib.formats.ED25519KEY_SCHEMA' format that
  # also includes the keyid.
  ed25519_key_metadata = securesystemslib.util.load_json_file(filepath)
  ed25519_key, junk = \
    securesystemslib.keys.format_metadata_to_key(ed25519_key_metadata)

  # Raise an exception if an unexpected key type is imported.  Redundant
  # validation of 'keytype'.  'securesystemslib.keys.format_metadata_to_key()'
  # should have fully validated 'ed25519_key_metadata'.
  if ed25519_key['keytype'] != 'ed25519': # pragma: no cover
    message = 'Invalid key type loaded: ' + repr(ed25519_key['keytype'])
    raise securesystemslib.exceptions.FormatError(message)

  return ed25519_key