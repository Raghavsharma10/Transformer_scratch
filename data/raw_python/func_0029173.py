def BNE(self, params):
        """
        BNE label

        Branch to the instruction at label if the Z flag is not set
        """
        label = self.get_one_parameter(self.ONE_PARAMETER, params)

        self.check_arguments(label_exists=(label,))

        # BNE label
        def BNE_func():
            if not self.is_Z_set():
                self.register['PC'] = self.labels[label]

        return BNE_func