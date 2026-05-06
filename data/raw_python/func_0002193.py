def parse_csv_dataset(data, handle_units):
    """Parse CSV data into a netCDF-like dataset."""
    fobj = BytesIO(data)
    names, units = parse_csv_header(fobj.readline().decode('utf-8'))
    arrs = np.genfromtxt(fobj, dtype=None, names=names, delimiter=',', unpack=True,
                         converters={'date': lambda s: parse_iso_date(s.decode('utf-8'))})
    d = {}
    for f in arrs.dtype.fields:
        dat = arrs[f]
        if dat.dtype == np.object:
            dat = dat.tolist()
        d[f] = handle_units(dat, units.get(f, None))
    return d