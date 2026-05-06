def WriteTo(self, values):
    """Writes values to a byte stream.

    Args:
      values (tuple[object, ...]): values to copy to the byte stream.

    Returns:
      bytes: byte stream.

    Raises:
      IOError: if byte stream cannot be written.
      OSError: if byte stream cannot be read.
    """
    try:
      return self._struct.pack(*values)
    except (TypeError, struct.error) as exception:
      raise IOError('Unable to write stream with error: {0!s}'.format(
          exception))