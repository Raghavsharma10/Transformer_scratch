def fpid_query(duration, fpdata, metadata=None):
    """Send fingerprint data to Last.fm to get the corresponding
    fingerprint ID, which can then be used to fetch metadata.
    duration is the length of the track in (integral) seconds.
    If metadata is provided, it is a dictionary with three optional
    fields reflecting the current metadata for the file: "artist",
    "album", and "title". These values are optional but might help
    improve the database. Returns the fpid, an integer, or raises a
    QueryError.
    """
    metadata = metadata or {}
    params = {
        'artist': metadata.get('artist', ''),
        'album': metadata.get('album', ''),
        'track': metadata.get('title', ''),
        'duration': duration,
    }
    url = '%s?%s' % (URL_FPID, urllib.urlencode(params))
    try:
        res = _query_wrap(formdata_post, url, {'fpdata': fpdata})
    except urllib2.HTTPError:
        raise CommunicationError('ID query failed')
    except httplib.BadStatusLine:
        raise CommunicationError('bad response in ID query')
    except IOError:
        raise CommunicationError('ID query failed')
    
    try:
        fpid, status = res.split()[:2]
        fpid = int(fpid)
    except ValueError:
        raise BadResponseError('malformed response: ' + res)

    if status == 'NEW':
        raise NotFoundError()
    elif status == 'FOUND':
        return fpid
    else:
        raise BadResponseError('unknown status: ' + res)