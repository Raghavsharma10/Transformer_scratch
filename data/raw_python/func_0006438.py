def get_global_register_bitsets(self, register_addresses):  # TOTO: add sorting
        """Calculating register bitsets from register addresses.

        Usage: get_global_register_bitsets([regaddress_1, regaddress_2, ...])
        Receives: list of register addresses
        Returns: list of register bitsets

        """
        register_bitsets = []
        for register_address in register_addresses:
            register_objects = self.get_global_register_objects(addresses=register_address)
            register_bitset = bitarray(16, endian='little')  # TODO remove hardcoded register size, see also below
            register_bitset.setall(0)
            register_littleendian = False
            for register_object in register_objects:
                if register_object['register_littleendian']:  # check for register endianness
                    register_littleendian = True
                if (16 * register_object['address'] + register_object['offset'] < 16 * (register_address + 1) and 16 * register_object['address'] + register_object['offset'] + register_object['bitlength'] > 16 * register_address):
                    reg = bitarray_from_value(value=register_object['value'], size=register_object['bitlength'])
                    if register_object['littleendian']:
                        reg.reverse()
# register_bitset[max(0, 16 * (register_object['address'] - register_address) + register_object['offset']):min(16, 16 * (register_object['address'] - register_address) + register_object['offset'] + register_object['bitlength'])] |= reg[max(0, 16 * (register_address - register_object['address']) - register_object['offset']):min(register_object['bitlength'], 16 * (register_address - register_object['address'] + 1) - register_object['offset'])]  # [ bit(n) bit(n-1)... bit(0) ]
                    register_bitset[max(0, 16 - 16 * (register_object['address'] - register_address) - register_object['offset'] - register_object['bitlength']):min(16, 16 - 16 * (register_object['address'] - register_address) - register_object['offset'])] |= reg[max(0, register_object['bitlength'] - 16 - 16 * (register_address - register_object['address']) + register_object['offset']):min(register_object['bitlength'], register_object['bitlength'] + 16 - 16 * (register_address - register_object['address'] + 1) + register_object['offset'])]  # [ bit(0)... bit(n-1) bit(n) ]
                else:
                    raise Exception("wrong register object")
            if register_littleendian:
                register_bitset.reverse()
            register_bitsets.append(register_bitset)
        return register_bitsets