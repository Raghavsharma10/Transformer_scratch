def _get_component(self, string, initial_pos):
    ''' given a string and a position, return both an updated position and
    either a Component Object or a String back to the caller '''
    add_code = string[initial_pos:initial_pos + self.ADDR_CODE_LENGTH]
    
    if add_code == 'REM':
      raise ish_reportException("This is a remarks record")
    if add_code == 'EQD':
      raise ish_reportException("This is EQD record")

    initial_pos += self.ADDR_CODE_LENGTH 
    try:
      useable_map = self.MAP[add_code]
    except:
      raise BaseException("Cannot find code %s in string %s (%d)." % (add_code, string, initial_pos))

    # if there is no defined length, then read next three chars to get it
    # this only applies to REM types, which have 3 chars for the type, then variable
    if useable_map[1] is False:
      chars_to_read = string[initial_pos + self.ADDR_CODE_LENGTH:initial_pos + \
                      (self.ADDR_CODE_LENGTH * 2)]
      chars_to_read = int(chars_to_read)
      initial_pos += (self.ADDR_CODE_LENGTH * 2)
    else:
      chars_to_read = useable_map[1]

    new_position = initial_pos + chars_to_read
    string_value = string[initial_pos:new_position]

    try:
      object_value = useable_map[2]()
      object_value.loads(string_value)
    except IndexError as err:
      object_value = string_value

    return (new_position, [add_code, object_value])