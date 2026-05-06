def _map_response(response, decode=False):
    """Maps a urllib3 response to a httplib/httplib2 Response."""
    # This causes weird deepcopy errors, so it's commented out for now.
    # item._urllib3_response = response
    item = httplib2.Response(response.getheaders())
    item.status = response.status
    item['status'] = str(item.status)
    item.reason = response.reason
    item.version = response.version

    # httplib2 expects the content-encoding header to be stripped and the
    # content length to be the length of the uncompressed content.
    # This does not occur for 'HEAD' requests.
    if decode and item.get('content-encoding') in ['gzip', 'deflate']:
        item['content-length'] = str(len(response.data))
        item['-content-encoding'] = item.pop('content-encoding')

    return item