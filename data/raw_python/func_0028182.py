def ReadFrom(self, byte_stream):
    """Read values from a byte stream.

    Args:
      byte_stream (bytes): byte stream.

    Returns:
      tuple[object, ...]: values copies from the byte stream.

    Raises:
      IOError: if byte stream cannot be read.
      OSError: if byte stream cannot be read.
    """
    try:
      return self._struct.unpack_from(byte_stream)
    except (TypeError, struct.error) as exception:
      raise IOError('Unable to read byte stream with error: {0!s}'.format(
          exception))