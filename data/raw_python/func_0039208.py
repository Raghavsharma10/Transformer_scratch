def usn_v4_record(header, record):
    """Extracts USN V4 record information."""
    length, major_version, minor_version = header
    fields = V4_RECORD.unpack_from(record, RECORD_HEADER.size)

    raise NotImplementedError('Not implemented')