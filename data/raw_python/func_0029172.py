def BMI(self, params):
        """
        BMI label

        Branch to the instruction at label if the N flag is set
        """
        label = self.get_one_parameter(self.ONE_PARAMETER, params)

        self.check_arguments(label_exists=(label,))

        # BMI label
        def BMI_func():
            if self.is_N_set():
                self.register['PC'] = self.labels[label]

        return BMI_func