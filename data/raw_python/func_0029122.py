def directive_SPACE(self, label, params):
        """
        label   SPACE num

        Allocate space on the stack. `num` is the number of bytes to allocate
        """
        # TODO allow equations

        params = params.strip()
        try:
            self.convert_to_integer(params)
        except ValueError:
            warnings.warn("Unknown parameters; {}".format(params))
            return

        self.labels[label] = self.space_pointer
        if params in self.equates:
            params = self.equates[params]
        self.space_pointer += self.convert_to_integer(params)