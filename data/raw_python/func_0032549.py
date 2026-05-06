def is_authenticode_signed(filename):
    """Returns True if the file is signed with authenticode"""
    with open(filename, 'rb') as fp:
        fp.seek(0)
        magic = fp.read(2)
        if magic != b'MZ':
            return False

        # First grab the pointer to the coff_header, which is at offset 60
        fp.seek(60)
        coff_header_offset = struct.unpack('<L', fp.read(4))[0]

        # Check the COFF magic
        fp.seek(coff_header_offset)
        magic = fp.read(4)
        if magic != b'PE\x00\x00':
            return False

        # Get the PE type
        fp.seek(coff_header_offset + 0x18)
        pe_type = struct.unpack('<h', fp.read(2))[0]

        if pe_type == 0x10b:
            # PE32 file (32-bit apps)
            number_of_data_dirs_offset = coff_header_offset + 0x74
        elif pe_type == 0x20b:
            # PE32+ files (64-bit apps)
            # PE32+ files have slightly larger fields in the header
            number_of_data_dirs_offset = coff_header_offset + 0x74 + 16
        else:
            return False

        fp.seek(number_of_data_dirs_offset)
        num_data_dirs = struct.unpack('<L', fp.read(4))[0]

        if num_data_dirs < 5:
            # Probably shouldn't happen, but just in case
            return False

        cert_table_offset = number_of_data_dirs_offset + 4*8 + 4
        fp.seek(cert_table_offset)
        addr, size = struct.unpack('<LL', fp.read(8))
        if not addr or not size:
            return False
        # Check that addr is inside the file
        fp.seek(addr)
        if fp.tell() != addr:
            return False
        cert = fp.read(size)
        if len(cert) != size:
            return False
        return True