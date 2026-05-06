def parse_description_xml(location):
    """ Extract serial number, base ip, and img url from description.xml

    missing data from XML returns AttributeError
    malformed XML returns ParseError

    Refer to included example for URLBase and serialNumber elements
    """
    class _URLBase(str):
        """ Convenient access to hostname (ip) portion of the URL """
        @property
        def hostname(self):
            return urlsplit(self).hostname

    # """TODO: review error handling on xml"""
    # may want to suppress ParseError in the event that it was caused
    # by a none bridge device although this seems unlikely
    try:
        xml_str = from_url(location)
    except urllib.request.HTTPError as error:
        logger.info("No description for %s: %s", location, error)
        return None, error
    except urllib.request.URLError as error:
        logger.info("No HTTP server for %s: %s", location, error)
        return None, error
    else:
        root = ET.fromstring(xml_str)
        rootname = {'root': root.tag[root.tag.find('{')+1:root.tag.find('}')]}
        baseip = root.find('root:URLBase', rootname).text
        device = root.find('root:device', rootname)
        serial = device.find('root:serialNumber', rootname).text
        # anicon = device.find('root:iconList', rootname).find('root:icon', rootname)
        # imgurl = anicon.find('root:url', rootname).text

        # Alternatively, could look directly in the modelDescription field
        if all(x in xml_str.lower() for x in ['philips', 'hue']):
            return serial, _URLBase(baseip)
        else:
            return None, None