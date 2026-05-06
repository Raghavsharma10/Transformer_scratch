def _is_valid_input(self, inpt, metadata, array):
        """The _is_valid_input method takes three arguments:

        the user input to be checked, the associated  osid.Metadata object
        containing validation requirements and a boolean value indicating
        whether this is an array value.

        """
        # pylint: disable=too-many-branches,no-self-use
        # Please redesign, and move to utility module
        syntax = metadata.get_syntax()

        # First check if this is a required data element
        if metadata.is_required and not inpt:
            return False

        valid = True  # Innocent until proven guilty
        # Recursively run through all the elements of an array
        if array:
            if len(inpt) < metadata.get_minimum_elements():
                valid = False
            elif len(inpt) > metadata.get_maximum_elements():
                valid = False
            else:
                for element in inpt:
                    valid = (valid and self._is_valid_input(element, metadata, False))

        # Run through all the possible syntax types
        elif syntax == 'ID':
            valid = self._is_valid_id(inpt)
        elif syntax == 'TYPE':
            valid = self._is_valid_type(inpt)
        elif syntax == 'BOOLEAN':
            valid = self._is_valid_boolean(inpt)
        elif syntax == 'STRING':
            valid = self._is_valid_string(inpt, metadata)
        elif syntax == 'INTEGER':
            valid = self._is_valid_integer(inpt, metadata)
        elif syntax == 'DECIMAL':
            valid = self._is_valid_decimal(inpt, metadata)
        elif syntax == 'DATETIME':
            valid = self._is_valid_date_time(inpt, metadata)
        elif syntax == 'DURATION':
            valid = self._is_valid_duration(inpt, metadata)
        elif syntax == 'CARDINAL':
            valid = self._is_valid_cardinal(inpt, metadata)
        elif syntax == 'INTEGER':
            valid = self._is_valid_integer(inpt, metadata)
        elif syntax == 'DECIMAL':
            valid = self._is_valid_decimal(inpt, metadata)
        else:
            raise errors.OperationFailed('no validation function available for ' + syntax)

        return valid