def directive_DCD(self, label, params):
        """
        label   DCD value[, value ...]

        Allocate a word space in read only memory for the value or list of values
        """
        # TODO make this read only
        # TODO check for param size
        # TODO can take any length comma separated values (VAL DCD 1, 0x2, 3, 4

        params = params.strip()
        try:
            self.convert_to_integer(params)
        except ValueError:
            # TODO allow word DCDs (like SP_INIT, Reset_Handler)
            warnings.warn("Cannot reserve constant words; {}".format(params))
            return

        # Align address
        if self.space_pointer % 4 != 0:
            self.space_pointer += self.space_pointer % 4
        self.labels[label] = self.space_pointer
        if params in self.equates:
            params = self.equates[params]
        for i in range(4):
            self.memory[self.space_pointer + i] = (self.convert_to_integer(params) >> (8*i)) & 0xFF
        self.space_pointer += 4