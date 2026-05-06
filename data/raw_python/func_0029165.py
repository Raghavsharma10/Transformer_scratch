def BGT(self, params):
        """
        BGT label

        Branch to the instruction at label if the N flag is the same as the V flag and the Z flag is not set
        """
        label = self.get_one_parameter(self.ONE_PARAMETER, params)

        self.check_arguments(label_exists=(label,))

        # BGT label
        def BGT_func():
            if (self.is_N_set() == self.is_V_set()) and not self.is_Z_set():
                self.register['PC'] = self.labels[label]

        return BGT_func