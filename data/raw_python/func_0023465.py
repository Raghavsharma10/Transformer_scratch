def compute_hash_info(fd, unit_size=None):
    """Get MediaFireHashInfo structure from the fd, unit_size

    fd -- file descriptor - expects exclusive access because of seeking
    unit_size -- size of a single unit

    Returns MediaFireHashInfo:
    hi.file -- sha256 of the whole file
    hi.units -- list of sha256 hashes for each unit
    """

    logger.debug("compute_hash_info(%s, unit_size=%s)", fd, unit_size)

    fd.seek(0, os.SEEK_END)
    file_size = fd.tell()
    fd.seek(0, os.SEEK_SET)

    units = []
    unit_counter = 0

    file_hash = hashlib.sha256()
    unit_hash = hashlib.sha256()

    for chunk in iter(lambda: fd.read(HASH_CHUNK_SIZE_BYTES), b''):
        file_hash.update(chunk)

        unit_hash.update(chunk)
        unit_counter += len(chunk)

        if unit_size is not None and unit_counter == unit_size:
            # flush the current unit hash
            units.append(unit_hash.hexdigest().lower())
            unit_counter = 0
            unit_hash = hashlib.sha256()

    if unit_size is not None and unit_counter > 0:
        # leftover block
        units.append(unit_hash.hexdigest().lower())

    fd.seek(0, os.SEEK_SET)

    return MediaFireHashInfo(
        file=file_hash.hexdigest().lower(),
        units=units,
        size=file_size
    )