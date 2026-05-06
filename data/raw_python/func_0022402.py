def compare(self, digest_2, is_hex = False):
        """
        returns difference between the nilsimsa digests between the current
        object and a given digest
        """
        # convert hex string to list of ints
        if is_hex:
            digest_2 = convert_hex_to_ints(digest_2)

        bit_diff = 0
        for i in range(len(self.digest)):
            bit_diff += POPC[self.digest[i] ^ digest_2[i]]           #computes the bit diff between the i'th position of the digests

        return 128 - bit_diff