def build_command(self, command_name, **kwargs):
        """build command from command_name and keyword values

        Returns
        -------
        command_bitvector : list
            List of bitarrays.

        Usage
        -----
        Receives: command name as defined inside xml file, key-value-pairs as defined inside bit stream filed for each command
        """
#         command_name = command_name.lower()
        command_bitvector = bitarray(0, endian='little')
        if command_name not in self.commands:
            raise ValueError('Unknown command %s' % command_name)
        command_object = self.commands[command_name]
        command_parts = re.split(r'\s*[+]\s*', command_object['bitstream'])
        # for index, part in enumerate(command_parts, start = 1): # loop over command parts
        for part in command_parts:  # loop over command parts
            try:
                command_part_object = self.commands[part]
            except KeyError:
                command_part_object = None
            if command_part_object and 'bitstream'in command_part_object:  # command parts of defined content and length, e.g. Slow, ...
                if string_is_binary(command_part_object['bitstream']):
                    command_bitvector += bitarray(command_part_object['bitstream'], endian='little')
                else:
                    command_bitvector += self.build_command(part, **kwargs)
            elif command_part_object:  # Command parts with any content of defined length, e.g. ChipID, Address, ...
                if part in kwargs:
                    value = kwargs[part]
                else:
                    raise ValueError('Value of command part %s not given' % part)
                try:
                    command_bitvector += value
                except TypeError:  # value is no bitarray
                    if string_is_binary(value):
                        value = int(value, 2)
                    try:
                        command_bitvector += bitarray_from_value(value=int(value), size=command_part_object['bitlength'], fmt='I')
                    except Exception:
                        raise TypeError("Type of value not supported")
            elif string_is_binary(part):
                command_bitvector += bitarray(part, endian='little')
            # elif part in kwargs.keys():
            #    command_bitvector += kwargs[command_name]
            else:
                raise ValueError("Cannot process command part %s" % part)
        if command_bitvector.length() != command_object['bitlength']:
            raise ValueError("Command has unexpected length")
        if command_bitvector.length() == 0:
            raise ValueError("Command has length 0")
        return command_bitvector