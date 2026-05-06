def _GetByteStreamOperation(self):
    """Retrieves the byte stream operation.

    Returns:
      ByteStreamOperation: byte stream operation or None if unable to determine.
    """
    byte_order_string = self.GetStructByteOrderString()
    format_string = self.GetStructFormatString()  # pylint: disable=assignment-from-none
    if not format_string:
      return None

    format_string = ''.join([byte_order_string, format_string])
    return byte_operations.StructOperation(format_string)