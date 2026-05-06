def parse_metadata(xml):
    """Given an XML document (string) returned from metadata_query(),
    parse the response into a list of track info dicts. May raise an
    APIError if the lookup fails.
    """
    try:
        root = etree.fromstring(xml)
    except (ExpatError, etree.ParseError):
        # The Last.fm API occasionally generates malformed XML when its
        # includes an illegal character (UTF8-legal but prohibited by
        # the XML standard).
        raise CommunicationError('malformed XML response')
    
    status = root.attrib['status']
    if status == 'failed':
        error = root.find('error')
        raise APIError(int(error.attrib['code']), error.text)
    
    out = []
    for track in root.find('tracks').findall('track'):
        out.append({
            'rank': float(track.attrib['rank']),
            'artist': track.find('artist').find('name').text,
            'artist_mbid': track.find('artist').find('mbid').text,
            'title': track.find('name').text,
            'track_mbid': track.find('mbid').text,
        })
    return out