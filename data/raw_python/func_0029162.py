def BCS(self, params):
        """
        BCS label

        Branch to the instruction at label if the C flag is set
        """
        label = self.get_one_parameter(self.ONE_PARAMETER, params)

        self.check_arguments(label_exists=(label,))

        # BCS label
        def BCS_func():
            if self.is_C_set():
                self.register['PC'] = self.labels[label]

        return BCS_func