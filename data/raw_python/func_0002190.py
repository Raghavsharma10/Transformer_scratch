def parse_xml_dataset(elem, handle_units):
    """Create a netCDF-like dataset from XML data."""
    points, units = zip(*[parse_xml_point(p) for p in elem.findall('point')])
    # Group points by the contents of each point
    datasets = {}
    for p in points:
        datasets.setdefault(tuple(p), []).append(p)

    all_units = combine_dicts(units)
    return [combine_xml_points(d, all_units, handle_units) for d in datasets.values()]