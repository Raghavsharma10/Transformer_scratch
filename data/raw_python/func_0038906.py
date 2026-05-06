def hash_filesystem(filesystem, hashtype='sha1'):
    """Utility function for running the files iterator at once.

    Returns a dictionary.

        {'/path/on/filesystem': 'file_hash'}

    """
    try:
        return dict(filesystem.checksums('/'))
    except RuntimeError:
        results = {}

        logging.warning("Error hashing disk %s contents, iterating over files.",
                        filesystem.disk_path)

        for path in filesystem.nodes('/'):
            try:
                regular = stat.S_ISREG(filesystem.stat(path)['mode'])
            except RuntimeError:
                continue  # unaccessible node

            if regular:
                try:
                    results[path] = filesystem.checksum(path, hashtype=hashtype)
                except RuntimeError:
                    logging.debug("Unable to hash %s.", path)

        return results