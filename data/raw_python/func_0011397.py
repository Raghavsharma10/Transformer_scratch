def reserve_bits(self, num_bits, stream):
        """Used to "reserve" ``num_bits`` amount of bits in order to keep track
        of consecutive bitfields (or are the called bitfield groups?).

        E.g. ::

            struct {
                char a:8, b:8;
                char c:4, d:4, e:8;
            }

        :param int num_bits: The number of bits to claim
        :param pfp.bitwrap.BitwrappedStream stream: The stream to reserve bits on
        :returns: If room existed for the reservation
        """
        padded = self.interp.get_bitfield_padded()
        num_bits = PYVAL(num_bits)

        if padded:
            num_bits = PYVAL(num_bits)
            if num_bits + self.reserved_bits > self.max_bits:
                return False

        # if unpadded, always allow it
        if not padded:
            if self._cls_bits is None:
                self._cls_bits = []

            # reserve bits will only be called just prior to reading the bits,
            # so check to see if we have enough bits in self._cls_bits, else
            # read what's missing
            diff = len(self._cls_bits) - num_bits
            if diff < 0:
                self._cls_bits += stream.read_bits(-diff)

        self.reserved_bits += num_bits
        return True