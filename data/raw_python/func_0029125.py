def directive_DCB(self, label, params):
        """
        label   DCB value[, value ...]

        Allocate a byte space in read only memory for the value or list of values
        """
        # TODO make this read only
        # TODO check for byte size
        self.labels[label] = self.space_pointer
        if params in self.equates:
            params = self.equates[params]
        self.memory[self.space_pointer] = self.convert_to_integer(params) & 0xFF
        self.space_pointer += 1