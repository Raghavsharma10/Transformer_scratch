def BVC(self, params):
        """
        BVC label

        Branch to the instruction at label if the V flag is not set
        """
        label = self.get_one_parameter(self.ONE_PARAMETER, params)

        self.check_arguments(label_exists=(label,))

        # BVC label
        def BVC_func():
            if not self.is_V_set():
                self.register['PC'] = self.labels[label]

        return BVC_func