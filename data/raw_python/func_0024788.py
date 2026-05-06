def is_valid_int(value):
        """Test if value can be rendered out of int."""
        if 0 <= value <= Parameter.MAX:  # This includes ON and OFF
            return True
        if value == Parameter.UNKNOWN_VALUE:
            return True
        if value == Parameter.CURRENT_POSITION:
            return True
        return False