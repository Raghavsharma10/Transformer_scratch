def parse_xml_station(elem):
    """Create a :class:`Station` instance from an XML tag."""
    stid = elem.attrib['id']
    name = elem.find('name').text
    lat = float(elem.find('latitude').text)
    lon = float(elem.find('longitude').text)
    elev = float(elem.find('elevation').text)
    return Station(id=stid, elevation=elev, latitude=lat, longitude=lon, name=name)