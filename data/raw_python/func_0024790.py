def from_percent(position_percent):
        """Create raw value out of percent position."""
        if not isinstance(position_percent, int):
            raise PyVLXException("Position::position_percent_has_to_be_int")
        if position_percent < 0:
            raise PyVLXException("Position::position_percent_has_to_be_positive")
        if position_percent > 100:
            raise PyVLXException("Position::position_percent_out_of_range")
        return bytes([position_percent*2, 0])