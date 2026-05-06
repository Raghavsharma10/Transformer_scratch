def write_catalog(filename, catalog, fmt=None, meta=None, prefix=None):
    """
    Write a catalog (list of sources) to a file with format determined by extension.

    Sources must be of type :class:`AegeanTools.models.OutputSource`,
    :class:`AegeanTools.models.SimpleSource`, or :class:`AegeanTools.models.IslandSource`.

    Parameters
    ----------
    filename : str
        Base name for file to write. `_simp`, `_comp`, or `_isle` will be added to differentiate
        the different types of sources that are being written.

    catalog : list
        A list of source objects. Sources must be of type :class:`AegeanTools.models.OutputSource`,
        :class:`AegeanTools.models.SimpleSource`, or :class:`AegeanTools.models.IslandSource`.

    fmt : str
        The file format extension.

    prefix : str
        Prepend each column name with "prefix_". Default is to prepend nothing.

    meta : dict
        A dictionary to be used as metadata for some file types (fits, VOTable).

    Returns
    -------
    None
    """
    if meta is None:
        meta = {}

    if prefix is None:
        pre=''
    else:
        pre = prefix + '_'

    def writer(filename, catalog, fmt=None):
        """
        construct a dict of the data
        this method preserves the data types in the VOTable
        """
        tab_dict = {}
        name_list = []
        for name in catalog[0].names:
            col_name = name
            if catalog[0].galactic:
                if name.startswith('ra'):
                    col_name = 'lon'+name[2:]
                elif name.endswith('ra'):
                    col_name = name[:-2] + 'lon'
                elif name.startswith('dec'):
                    col_name = 'lat'+name[3:]
                elif name.endswith('dec'):
                    col_name = name[:-3] + 'lat'
            col_name = pre + col_name
            tab_dict[col_name] = [getattr(c, name, None) for c in catalog]
            name_list.append(col_name)
        t = Table(tab_dict, meta=meta)
        # re-order the columns
        t = t[[n for n in name_list]]

        if fmt is not None:
            if fmt in ["vot", "vo", "xml"]:
                vot = from_table(t)
                # description of this votable
                vot.description = repr(meta)
                writetoVO(vot, filename)
            elif fmt in ['hdf5']:
                t.write(filename, path='data', overwrite=True)
            elif fmt in ['fits']:
                writeFITSTable(filename, t)
            else:
                ascii.write(t, filename, fmt, overwrite=True)
        else:
            ascii.write(t, filename, overwrite=True)
        return

    # sort the sources into types and then write them out individually
    components, islands, simples = classify_catalog(catalog)

    if len(components) > 0:
        new_name = "{1}{0}{2}".format('_comp', *os.path.splitext(filename))
        writer(new_name, components, fmt)
        log.info("wrote {0}".format(new_name))
    if len(islands) > 0:
        new_name = "{1}{0}{2}".format('_isle', *os.path.splitext(filename))
        writer(new_name, islands, fmt)
        log.info("wrote {0}".format(new_name))
    if len(simples) > 0:
        new_name = "{1}{0}{2}".format('_simp', *os.path.splitext(filename))
        writer(new_name, simples, fmt)
        log.info("wrote {0}".format(new_name))
    return