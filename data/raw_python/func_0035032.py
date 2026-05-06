def state():
    '''Get The playback state: 'playing', 'paused', or 'stopped'.

    If PLAYING or PAUSED, show information on current track.

    Calls PlaybackController.get_state(), and if state is PLAYING or PAUSED, get
      PlaybackController.get_current_track() and
      PlaybackController.get_time_position()'''

    server = getServer()
    state = server.core.playback.get_state()
    logging.debug('Got playback state: %r', state)
    if state.upper() == 'STOPPED':
        print('Playback is currently stopped')
    else:
        track = server.core.playback.get_current_track()
        logging.debug('Track is %r', track)
        logging.debug('Track loaded is %r', jsonrpclib.jsonclass.load(track))
        pos = server.core.playback.get_time_position()
        logging.debug('Pos is %r', pos)
        print('{} track: "{}", by {} (at {})'.format(state.title(),
                                                     track['name'],
                                                     ','.join([a['name'] for a in track['artists']]),
                                                     formatTimeposition(pos))
              )