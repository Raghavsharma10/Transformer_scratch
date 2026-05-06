def B(self, params):
        """
        B label

        Unconditional branch to the address at label
        """
        label = self.get_one_parameter(self.ONE_PARAMETER, params)

        self.check_arguments(label_exists=(label,))
        # TODO check if label is within +- 2 KB

        # B label
        def B_func():
            if label == '.':
                raise iarm.exceptions.EndOfProgram("You have reached an infinite loop")
            self.register['PC'] = self.labels[label]

        return B_func