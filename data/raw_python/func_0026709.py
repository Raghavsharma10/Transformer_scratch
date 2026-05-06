def compress_gz(fname):
    """Compress the file with the given name and delete the uncompressed file.

    The compressed filename is simply the input filename with '.gz' appended.

    Arguments
    ---------
    fname : str
        Name of the file to compress and delete.

    Returns
    -------
    comp_fname : str
        Name of the compressed file produced.  Equal to `fname + '.gz'`.

    """
    import shutil
    import gzip
    comp_fname = fname + '.gz'
    with codecs.open(fname, 'rb') as f_in, gzip.open(
            comp_fname, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(fname)
    return comp_fname