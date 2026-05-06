def read_bits(self, stream, num_bits, padded, left_right, endian):
        """Return ``num_bits`` bits, taking into account endianness and 
        left-right bit directions
        """
        if self._cls_bits is None and padded:
            raw_bits = stream.read_bits(self.cls.width*8)
            self._cls_bits = self._endian_transform(raw_bits, endian)

        if self._cls_bits is not None:
            if num_bits > len(self._cls_bits):
                raise errors.PfpError("BitfieldRW reached invalid state")

            if left_right:
                res = self._cls_bits[:num_bits]
                self._cls_bits = self._cls_bits[num_bits:]
            else:
                res = self._cls_bits[-num_bits:]
                self._cls_bits = self._cls_bits[:-num_bits]

            return res

        else:
            return stream.read_bits(num_bits)