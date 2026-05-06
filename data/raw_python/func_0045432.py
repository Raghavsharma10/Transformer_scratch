def compute(dvd_path):
    """Computes a Windows API IDvdInfo2::GetDiscID-compatible 64-bit Cyclic Redundancy Check
       checksum from the DVD .vob, .ifo and .bup files found in the supplied DVD path.
    """

    _check_dvd_path_exists(dvd_path)

    _check_video_ts_path_exists(dvd_path)

    # the polynomial used for this CRC-64 checksum is:
    # x^63 + x^60 + x^57 + x^55 + x^54 + x^50 + x^49 + x^46 + x^41 + x^38 + x^37 + x^34 + x^32 +
    # x^31 + x^30 + x^28 + x^25 + x^24 + x^21 + x^16 + x^13 + x^12 + x^11 + x^8 + x^7 + x^5 + x^2
    calculator = _Crc64Calculator(0x92c64265d32139a4)

    for video_ts_file_path in _get_video_ts_file_paths(dvd_path):
        calculator.update(_get_file_creation_time(video_ts_file_path))
        calculator.update(_get_file_size(video_ts_file_path))
        calculator.update(_get_file_name(video_ts_file_path))

    calculator.update(_get_vmgi_file_content(dvd_path))
    calculator.update(_get_vts01i_file_content(dvd_path))

    return calculator.crc64