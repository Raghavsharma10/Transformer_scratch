def _write_int(fname, data, append=True):
    """Write data to CSV file with validation."""
    # pylint: disable=W0705
    data_ex = pexdoc.exh.addex(ValueError, "There is no data to save to file")
    fos_ex = pexdoc.exh.addex(
        OSError, "File *[fname]* could not be created: *[reason]*"
    )
    data_ex((len(data) == 0) or ((len(data) == 1) and (len(data[0]) == 0)))
    try:
        pmisc.make_dir(fname)
        mode = "w" if append is False else "a"
        if sys.hexversion < 0x03000000:  # pragma: no cover, no branch
            with open(fname, mode) as file_handle:
                csv.writer(file_handle, delimiter=",").writerows(data)
        else:  # pragma: no cover
            with open(fname, mode, newline="") as file_handle:
                csv.writer(file_handle, delimiter=",").writerows(data)
    except (IOError, OSError) as eobj:
        fos_ex(True, _MF("fname", fname, "reason", eobj.strerror))