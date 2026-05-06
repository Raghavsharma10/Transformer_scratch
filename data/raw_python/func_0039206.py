def parse_record(header, record):
    """Parses a record according to its version."""
    major_version = header[1]

    try:
        return RECORD_PARSER[major_version](header, record)
    except (KeyError, struct.error) as error:
        raise RuntimeError("Corrupted USN Record") from error