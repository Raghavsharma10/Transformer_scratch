def parse_portal_json():
    """ Extract id, ip from https://www.meethue.com/api/nupnp

    Note: the ip is only the base and needs xml file appended, and
    the id is not exactly the same as the serial number in the xml
    """
    try:
        json_str = from_url('https://www.meethue.com/api/nupnp')
    except urllib.request.HTTPError as error:
        logger.error("Problem at portal: %s", error)
        raise
    except urllib.request.URLError as error:
        logger.warning("Problem reaching portal: %s", error)
        return []
    else:
        portal_list = []
        json_list = json.loads(json_str)
        for bridge in json_list:
            serial = bridge['id']
            baseip = bridge['internalipaddress']
            # baseip should look like "192.168.0.1"
            xmlurl = _build_from(baseip)
            # xmlurl should look like "http://192.168.0.1/description.xml"
            portal_list.append((serial, xmlurl))
        return portal_list