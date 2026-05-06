def is_valid(self):
        """Tests if ths form is in a valid state for submission.

        A form is valid if all required data has been supplied compliant
        with any constraints.

        return: (boolean) - false if there is a known error in this
                form, true otherwise
        raise:  OperationFailed - attempt to perform validation failed
        compliance: mandatory - This method must be implemented.

        """
        validity = True
        for element in self._validity_map:
            if self._validity_map[element] is not VALID:
                validity = False
        return validity