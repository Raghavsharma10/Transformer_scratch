def BGE(self, params):
        """
        BGE label

        Branch to the instruction at label if the N flag is the same as the V flag
        """
        label = self.get_one_parameter(self.ONE_PARAMETER, params)

        self.check_arguments(label_exists=(label,))

        # BGE label
        def BGE_func():
            if self.is_N_set() == self.is_V_set():
                self.register['PC'] = self.labels[label]

        return BGE_func