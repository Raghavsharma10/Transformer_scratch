def from_raw(raw):
        """Test if raw packets are valid for initialization of Position."""
        if not isinstance(raw, bytes):
            raise PyVLXException("Position::raw_must_be_bytes")
        if len(raw) != 2:
            raise PyVLXException("Position::raw_must_be_two_bytes")
        if raw != Position.from_int(Position.CURRENT_POSITION) and \
                raw != Position.from_int(Position.UNKNOWN_VALUE) and \
                Position.to_int(raw) > Position.MAX:
            raise PyVLXException("position::raw_exceed_limit", raw=raw)
        return raw