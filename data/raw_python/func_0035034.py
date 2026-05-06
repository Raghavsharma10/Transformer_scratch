def tracklist():
    '''Get tracklist

    Calls TracklistController.get_tl_tracks()
    '''
    _c = 0
    server = getServer()
    _current = server.core.tracklist.index()
    for t in server.core.tracklist.get_tl_tracks():
        logging.debug('Got tl trak: %r', t)
        currently = ' -- CURRENT' if t['tlid'] == _current else ''
        print('{}: {}{}'.format(t['tlid'], t['track']['name'], currently))
        _c = _c+1
    print('==='*6)
    print('{} tracks in tracklist'.format(_c))