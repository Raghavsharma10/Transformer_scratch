def metadata_query(fpid, apikey):
    """Queries the Last.fm servers for metadata about a given
    fingerprint ID (an integer). Returns the XML response (a string).
    """
    params = {
        'method': 'track.getFingerprintMetadata',
        'fingerprintid': fpid,
        'api_key': apikey,
    }
    url = '%s?%s' % (URL_METADATA, urllib.urlencode(params))
    try:
        fh = _query_wrap(urllib.urlopen, url)
    except urllib2.HTTPError:
        raise CommunicationError('metadata query failed')
    except httplib.BadStatusLine:
        raise CommunicationError('bad response in metadata query')
    except IOError:
        raise CommunicationError('metadata query failed')
    return fh.read()