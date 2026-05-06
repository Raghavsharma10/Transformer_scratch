def directive_DCH(self, label, params):
        """
        label   DCH value[, value ...]

        Allocate a half word space in read only memory for the value or list of values
        """
        # TODO make this read only
        # TODO check for word size

        # Align address
        if self.space_pointer % 2 != 0:
            self.space_pointer += self.space_pointer % 2
        self.labels[label] = self.space_pointer
        if params in self.equates:
            params = self.equates[params]
        for i in range(2):
            self.memory[self.space_pointer + i] = (self.convert_to_integer(params) >> (8 * i)) & 0xFF
        self.space_pointer += 2