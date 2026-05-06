def dearmor(text, verify=True):
    """
    Given a string in ASCII Armor format, returns the decoded binary data.
    If verify=True (the default), the CRC is decoded and checked against that
    of the decoded data, otherwise it is ignored. If the checksum does not
    match, a BadChecksumError exception is raised.
    """
    lines = text.strip().split('\n')
    data_lines = []
    check_data = None
    started = False
    in_body = False
    for line in lines:
        if line.startswith('-----BEGIN'):
            started = True
        elif line.startswith('-----END'):
            break
        elif started:
            if in_body:
                if line.startswith('='):
                    # Once we get the checksum data, we're done.
                    check_data = line[1:5].encode('ascii')
                    break
                else:
                    # This is part of the base64-encoded data.
                    data_lines.append(line)
            else:
                if line.strip():
                    # This is a header line, which we basically ignore for now.
                    pass
                else:
                    # The data starts after an empty line.
                    in_body = True
    b64_str = ''.join(data_lines)
    # Python 3's b64decode expects bytes, not a string. We know base64 is ASCII, though.
    data = base64.b64decode(b64_str.encode('ascii'))
    if verify and check_data:
        # The 24-bit CRC is in big-endian, so we add a null byte to the beginning.
        crc = struct.unpack('>L', b'\0' + base64.b64decode(check_data))[0]
        if crc != crc24(data):
            raise BadChecksumError()
    return data