def _is_valid_string(self, inpt, metadata):
        """Checks if input is a valid string"""
        if not is_string(inpt):
            return False
        if metadata.get_minimum_string_length() and len(inpt) < metadata.get_minimum_string_length():
            return False
        elif metadata.get_maximum_string_length() and len(inpt) > metadata.get_maximum_string_length():
            return False
        if metadata.get_string_set() and inpt not in metadata.get_string_set():
            return False
        else:
            return True