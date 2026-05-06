def BHI(self, params):
        """
        BHI label

        Branch to the instruction at label if the C flag is set and the Z flag is not set
        """
        label = self.get_one_parameter(self.ONE_PARAMETER, params)

        self.check_arguments(label_exists=(label,))

        # BHI label
        def BHI_func():
            if self.is_C_set() and not self.is_Z_set():
                self.register['PC'] = self.labels[label]

        return BHI_func