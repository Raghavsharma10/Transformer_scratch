def frame_from_raw(raw):
    """Create and return frame from raw bytes."""
    command, payload = extract_from_frame(raw)
    frame = create_frame(command)
    if frame is None:
        PYVLXLOG.warning("Command %s not implemented, raw: %s", command, ":".join("{:02x}".format(c) for c in raw))
        return None
    frame.validate_payload_len(payload)
    frame.from_payload(payload)
    return frame