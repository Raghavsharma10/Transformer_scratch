def parse_station_table(root):
    """Parse station list XML file."""
    stations = [parse_xml_station(elem) for elem in root.findall('station')]
    return {st.id: st for st in stations}