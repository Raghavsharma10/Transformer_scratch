def _get_remarks_component(self, string, initial_pos):
    ''' Parse the remarks into the _remarks dict '''
    remarks_code = string[initial_pos:initial_pos + self.ADDR_CODE_LENGTH]
    if remarks_code != 'REM':
      raise ish_reportException("Parsing remarks. Expected REM but got %s." % (remarks_code,))

    expected_length = int(string[0:4]) + self.PREAMBLE_LENGTH
    position = initial_pos + self.ADDR_CODE_LENGTH
    while position < expected_length:
      key = string[position:position + self.ADDR_CODE_LENGTH]
      if key == 'EQD':
        break
      chars_to_read = string[position + self.ADDR_CODE_LENGTH:position + \
                      (self.ADDR_CODE_LENGTH * 2)]
      chars_to_read = int(chars_to_read)
      position += (self.ADDR_CODE_LENGTH * 2)
      string_value = string[position:position + chars_to_read]
      self._remarks[key] = string_value
      position += chars_to_read