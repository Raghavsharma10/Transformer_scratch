def digests_are_equal(digest1, digest2):
  """
  <Purpose>
    While protecting against timing attacks, compare the hexadecimal arguments
    and determine if they are equal.

  <Arguments>
    digest1:
      The first hexadecimal string value to compare.

    digest2:
      The second hexadecimal string value to compare.

  <Exceptions>
    securesystemslib.exceptions.FormatError: If the arguments are improperly
    formatted.

  <Side Effects>
    None.

  <Return>
    Return True if 'digest1' is equal to 'digest2', False otherwise.
  """

  # Ensure the arguments have the appropriate number of objects and object
  # types, and that all dict keys are properly named.
  # Raise 'securesystemslib.exceptions.FormatError' if there is a mismatch.
  securesystemslib.formats.HEX_SCHEMA.check_match(digest1)
  securesystemslib.formats.HEX_SCHEMA.check_match(digest2)

  if len(digest1) != len(digest2):
    return False

  are_equal = True

  for element in range(len(digest1)):
    if digest1[element] != digest2[element]:
      are_equal = False

  return are_equal