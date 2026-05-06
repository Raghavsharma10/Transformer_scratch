def from_int(value):
        """Create raw out of position vlaue."""
        if not isinstance(value, int):
            raise PyVLXException("value_has_to_be_int")
        if not Parameter.is_valid_int(value):
            raise PyVLXException("value_out_of_range")
        return bytes([value >> 8 & 255, value & 255])