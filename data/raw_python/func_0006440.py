def get_pixel_register_bitset(self, register_object, bit_no, dc_no):
        """Calculating pixel register bitsets from pixel register addresses.

        Usage: get_pixel_register_bitset(object, bit_number, double_column_number)
        Receives: register object, bit number, double column number
        Returns: double column bitset

        """
        if not 0 <= dc_no < 40:
            raise ValueError("Pixel register %s: DC out of range" % register_object['name'])
        if not 0 <= bit_no < register_object['bitlength']:
            raise ValueError("Pixel register %s: bit number out of range" % register_object['name'])
        col0 = register_object['value'][dc_no * 2, :]
        sel0 = (2 ** bit_no == (col0 & 2 ** bit_no))
        bv0 = bitarray(sel0.tolist(), endian='little')
        col1 = register_object['value'][dc_no * 2 + 1, :]
        sel1 = (2 ** bit_no == (col1 & 2 ** bit_no))
        # sel1 = sel1.astype(numpy.uint8) # copy of array
        # sel1 = sel1.view(dtype=np.uint8) # in-place type conversion
        bv1 = bitarray(sel1.tolist(), endian='little')
        bv1.reverse()  # shifted first
        # bv = bv1+bv0
        return bv1 + bv0