def BEQ(self, params):
        """
        BEQ label

        Branch to the instruction at label if the Z flag is set
        """
        label = self.get_one_parameter(self.ONE_PARAMETER, params)

        self.check_arguments(label_exists=(label,))

        # BEQ label
        def BEQ_func():
            if self.is_Z_set():
                self.register['PC'] = self.labels[label]

        return BEQ_func