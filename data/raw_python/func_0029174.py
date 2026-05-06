def BPL(self, params):
        """
        BPL label

        Branch to the instruction at label if the N flag is set
        """
        label = self.get_one_parameter(self.ONE_PARAMETER, params)

        self.check_arguments(label_exists=(label,))

        # BPL label
        def BPL_func():
            if not self.is_N_set():
                self.register['PC'] = self.labels[label]

        return BPL_func