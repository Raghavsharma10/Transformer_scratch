def open_file(name, mode=None, driver=None, libver=None, userblock_size=None, **kwargs):
    """Open an ARF file, creating as necessary.

    Use this instead of h5py.File to ensure that root-level attributes and group
    creation property lists are set correctly.

    """
    import sys
    import os
    from h5py import h5p
    from h5py._hl import files

    try:
        # If the byte string doesn't match the default
        # encoding, just pass it on as-is.  Note Unicode
        # objects can always be encoded.
        name = name.encode(sys.getfilesystemencoding())
    except (UnicodeError, LookupError):
        pass
    exists = os.path.exists(name)
    try:
        fcpl = h5p.create(h5p.FILE_CREATE)
        fcpl.set_link_creation_order(
            h5p.CRT_ORDER_TRACKED | h5p.CRT_ORDER_INDEXED)
    except AttributeError:
        # older version of h5py
        fp = files.File(name, mode=mode, driver=driver,
                        libver=libver, **kwargs)
    else:
        fapl = files.make_fapl(driver, libver, **kwargs)
        fp = files.File(files.make_fid(name, mode, userblock_size, fapl, fcpl))

    if not exists and fp.mode == 'r+':
        set_attributes(fp,
                       arf_library='python',
                       arf_library_version=__version__,
                       arf_version=spec_version)
    return fp