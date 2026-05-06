def save_catalog(filename, catalog, meta=None, prefix=None):
    """
    Save a catalogue of sources using filename as a model. Meta data can be written to some file types
    (fits, votable).

    Each type of source will be in a separate file:

    - base_comp.ext :class:`AegeanTools.models.OutputSource`
    - base_isle.ext :class:`AegeanTools.models.IslandSource`
    - base_simp.ext :class:`AegeanTools.models.SimpleSource`


    Where filename = `base.ext`

    Parameters
    ----------
    filename : str
        Name of file to write, format is determined by extension.

    catalog : list
        A list of sources to write. Sources must be of type :class:`AegeanTools.models.OutputSource`,
        :class:`AegeanTools.models.SimpleSource`, or :class:`AegeanTools.models.IslandSource`.

    prefix : str
        Prepend each column name with "prefix_". Default is to prepend nothing.

    meta : dict
        Meta data to be written to the output file. Support for metadata depends on file type.

    Returns
    -------
    None
    """
    ascii_table_formats = {'csv': 'csv', 'tab': 'tab', 'tex': 'latex', 'html': 'html'}
    # .ann and .reg are handled by me
    meta = update_meta_data(meta)
    extension = os.path.splitext(filename)[1][1:].lower()
    if extension in ['ann', 'reg']:
        writeAnn(filename, catalog, extension)
    elif extension in ['db', 'sqlite']:
        writeDB(filename, catalog, meta)
    elif extension in ['hdf5', 'fits', 'vo', 'vot', 'xml']:
        write_catalog(filename, catalog, extension, meta, prefix=prefix)
    elif extension in ascii_table_formats.keys():
        write_catalog(filename, catalog, fmt=ascii_table_formats[extension], meta=meta, prefix=prefix)
    else:
        log.warning("extension not recognised {0}".format(extension))
        log.warning("You get tab format")
        write_catalog(filename, catalog, fmt='tab', prefix=prefix)
    return