def BLS(self, params):
        """
        BLS label

        Branch to the instruction at label if the C flag is not set or the Z flag is set
        """
        label = self.get_one_parameter(self.ONE_PARAMETER, params)

        self.check_arguments(label_exists=(label,))

        # BLS label
        def BLS_func():
            if (not self.is_C_set()) or self.is_Z_set():
                self.register['PC'] = self.labels[label]

        return BLS_func