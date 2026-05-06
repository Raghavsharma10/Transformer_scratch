def convert_representation(self, i):
        """
        Return the proper representation for the given integer
        """
        if self.number_representation == 'unsigned':
            return i
        elif self.number_representation == 'signed':
            if i & (1 << self.interpreter._bit_width - 1):
                return -((~i + 1) & (2**self.interpreter._bit_width - 1))
            else:
                return i
        elif self.number_representation == 'hex':
            return hex(i)