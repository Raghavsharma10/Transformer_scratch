def BLT(self, params):
        """
        BLT label

        Branch to the instruction at label if the N flag is not the same as the V flag
        """
        label = self.get_one_parameter(self.ONE_PARAMETER, params)

        self.check_arguments(label_exists=(label,))

        # BLT label
        def BLT_func():
            if self.is_N_set() != self.is_V_set():
                self.register['PC'] = self.labels[label]

        return BLT_func