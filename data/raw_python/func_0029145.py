def BL(self, params):
        """
        BL label

        Branch to the label, storing the next instruction in the Link Register
        """
        label = self.get_one_parameter(self.ONE_PARAMETER, params)

        self.check_arguments(label_exists=(label,))
        # TODO check if label is within +- 16 MB

        # BL label
        def BL_func():
            self.register['LR'] = self.register['PC']  # No need for the + 1, PC already points to the next instruction
            self.register['PC'] = self.labels[label]

        return BL_func