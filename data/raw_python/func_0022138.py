def digest(self):
        """Get digest of data seen thus far as a list of bytes."""
        total = 0                           # number of triplets seen
        if self.count == 3:                 # 3 chars = 1 triplet
            total = 1
        elif self.count == 4:               # 4 chars = 4 triplets
            total = 4
        elif self.count > 4:                # otherwise 8 triplets/char less
            total = 8 * self.count - 28     # 28 'missed' during 'ramp-up'

        threshold = total / 256             # threshold for accumulators, using the mean

        code = [0]*32                       # start with all zero bits
        for i in range(256):                # for all 256 accumulators
            if self.acc[i] > threshold:     # if it meets the threshold
                code[i >> 3] += 1 << (i&7)  # set corresponding digest bit, equivalent to i/8, 2 ** (i % 8)

        return code[::-1]