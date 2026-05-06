def _construct_lookup_table(self, polynomial):
        """Precomputes a CRC-64 lookup table seeded from the supplied polynomial.
           No return value.
        """

        self._lookup_table = []

        for i in range(0, 256):
            lookup_value = i

            for _ in range(0, 8):
                if lookup_value & 0x1 == 0x1:
                    lookup_value = (lookup_value >> 1) ^ polynomial

                else:
                    lookup_value = lookup_value >> 1

            self._lookup_table.append(lookup_value)