def usn_v2_record(header, record):
    """Extracts USN V2 record information."""
    length, major_version, minor_version = header
    fields = V2_RECORD.unpack_from(record, RECORD_HEADER.size)

    return UsnRecord(length,
                     float('{}.{}'.format(major_version, minor_version)),
                     fields[0] | fields[1] << 16,  # 6 bytes little endian mft
                     fields[2],  # 2 bytes little endian mft sequence
                     fields[3] | fields[4] << 16,  # 6 bytes little endian mft
                     fields[5],  # 2 bytes little endian mft sequence
                     fields[6],
                     (datetime(1601, 1, 1) +
                      timedelta(microseconds=(fields[7] / 10))).isoformat(' '),
                     unpack_flags(fields[8], REASONS),
                     unpack_flags(fields[9], SOURCEINFO),
                     fields[10],
                     unpack_flags(fields[11], ATTRIBUTES),
                     str(struct.unpack_from('{}s'.format(fields[12]).encode(),
                                            record, fields[13])[0], 'utf16'))