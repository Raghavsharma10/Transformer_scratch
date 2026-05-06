def BLE(self, params):
        """
        BLE label

        Branch to the instruction at label if the Z flag is set or if the N flag is not the same as the V flag
        """
        label = self.get_one_parameter(self.ONE_PARAMETER, params)

        self.check_arguments(label_exists=(label,))

        # BLE label
        def BLE_func():
            if self.is_Z_set() or (self.is_N_set() != self.is_V_set()):
                self.register['PC'] = self.labels[label]

        return BLE_func