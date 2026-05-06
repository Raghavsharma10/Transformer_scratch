def parse_xml_point(elem):
    """Parse an XML point tag."""
    point = {}
    units = {}
    for data in elem.findall('data'):
        name = data.get('name')
        unit = data.get('units')
        point[name] = float(data.text) if name != 'date' else parse_iso_date(data.text)
        if unit:
            units[name] = unit
    return point, units