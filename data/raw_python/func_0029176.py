def BVS(self, params):
        """
        BVS label

        Branch to the instruction at label if the V flag is set
        """
        label = self.get_one_parameter(self.ONE_PARAMETER, params)

        self.check_arguments(label_exists=(label,))

        # BVS label
        def BVS_func():
            if self.is_V_set():
                self.register['PC'] = self.labels[label]

        return BVS_func