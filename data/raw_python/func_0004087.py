def guess_file_encoding(fh, default=DEFAULT_ENCODING):
    """Guess encoding from a file handle."""
    start = fh.tell()
    detector = chardet.UniversalDetector()
    while True:
        data = fh.read(1024 * 10)
        if not data:
            detector.close()
            break
        detector.feed(data)
        if detector.done:
            break

    fh.seek(start)
    return normalize_result(detector.result, default=default)