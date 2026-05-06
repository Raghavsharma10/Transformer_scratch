def pause():
    '''Pause playback.

    Calls PlaybackController.pause()'''

    server = getServer()
    server.core.playback.pause()
    pos = server.core.playback.get_time_position()
    print('Paused at {}'.format(formatTimeposition(pos)))