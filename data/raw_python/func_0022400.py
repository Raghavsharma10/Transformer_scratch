def compute_digest(self):
        """
        using a threshold (mean of the accumulator), computes the nilsimsa digest
        """
        num_trigrams = 0
        if self.num_char == 3:          # 3 chars -> 1 trigram
            num_trigrams = 1
        elif self.num_char == 4:        # 4 chars -> 4 trigrams
            num_trigrams = 4
        elif self.num_char > 4:         # > 4 chars -> 8 for each char
            num_trigrams = 8 * self.num_char - 28

        # threshhold is the mean of the acc buckets
        threshold = num_trigrams / 256.0

        digest = [0] * 32
        for i in range(256):
            if self.acc[i] > threshold:
                digest[i >> 3] += 1 << (i & 7)      # equivalent to i/8, 2**(i mod 7)

        self._digest = digest[::-1]