def BCC(self, params):
        """
        BCC label

        Branch to the instruction at label if the C flag is not set
        """
        label = self.get_one_parameter(self.ONE_PARAMETER, params)

        self.check_arguments(label_exists=(label,))

        # BCC label
        def BCC_func():
            if not self.is_C_set():
                self.register['PC'] = self.labels[label]

        return BCC_func