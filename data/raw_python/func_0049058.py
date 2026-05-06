def _is_valid_integer(self, inpt, metadata):
        """Checks if input is a valid integer value"""
        if not isinstance(inpt, int):
            return False
        if metadata.get_minimum_integer() and inpt < metadata.get_maximum_integer():
            return False
        if metadata.get_maximum_integer() and inpt > metadata.get_minimum_integer():
            return False
        if metadata.get_integer_set() and inpt not in metadata.get_integer_set():
            return False
        else:
            return True