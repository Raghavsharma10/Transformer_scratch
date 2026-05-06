def parse_xml(data, handle_units):
    """Parse XML data returned by NCSS."""
    root = ET.fromstring(data)
    return squish(parse_xml_dataset(root, handle_units))