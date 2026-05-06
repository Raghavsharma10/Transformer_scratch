def import_ecdsa_publickey_from_file(filepath):
  """
  <Purpose>
    Load the ECDSA public key object (conformant to
    'securesystemslib.formats.KEY_SCHEMA') stored in 'filepath'.  Return
    'filepath' in securesystemslib.formats.ECDSAKEY_SCHEMA format.

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
    An ECDSA key object conformant to
    'securesystemslib.formats.ECDSAKEY_SCHEMA'.
  """

  # Does 'filepath' have the correct format?
  # Ensure the arguments have the appropriate number of objects and object
  # types, and that all dict keys are properly named.
  # Raise 'securesystemslib.exceptions.FormatError' if there is a mismatch.
  securesystemslib.formats.PATH_SCHEMA.check_match(filepath)

  # ECDSA key objects are saved in json and metadata format.  Return the
  # loaded key object in securesystemslib.formats.ECDSAKEY_SCHEMA' format that
  # also includes the keyid.
  ecdsa_key_metadata = securesystemslib.util.load_json_file(filepath)
  ecdsa_key, junk = \
    securesystemslib.keys.format_metadata_to_key(ecdsa_key_metadata)

  # Raise an exception if an unexpected key type is imported.  Redundant
  # validation of 'keytype'.  'securesystemslib.keys.format_metadata_to_key()'
  # should have fully validated 'ecdsa_key_metadata'.
  if ecdsa_key['keytype'] != 'ecdsa-sha2-nistp256': # pragma: no cover
    message = 'Invalid key type loaded: ' + repr(ecdsa_key['keytype'])
    raise securesystemslib.exceptions.FormatError(message)

  return ecdsa_key