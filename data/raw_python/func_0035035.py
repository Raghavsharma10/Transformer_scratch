def play_backend_uri(argv=None):
    '''Get album or track from backend uri and play all tracks found.

    uri is a string which represents some directory belonging to a backend.

    Calls LibraryController.browse(uri) to get an album and LibraryController.lookup(uri)
    to get track'''

    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser(description='Browse directories and tracks at the given uri and play them/it.')
    parser.add_argument('uri',
                        help='The key that represents some directory belonging to a backend. E.g. plex:album:2323 or spotify:album:xxxx')
    parser.add_argument('-l', '--loglevel', help='Logging level. Default: %(default)s.',
        choices=('debug', 'info', 'warning', 'error'), default='warning')
    args = parse_args_and_apply_logging_level(parser, argv)
    server = getServer()
    hits = server.core.library.browse(args.uri)
    # browse(): Returns a list of mopidy.models.Ref objects for the directories and tracks at the given uri.
    logging.info('Got hits from browse(): %r', hits)
    if len(hits) == 0:
        # try track lookup
        hits = server.core.library.lookup(args.uri)
        logging.info('Got hits from lookup() : %r', hits)

    if len(hits) == 0:
        print('No hits for "{}"'.format(args.uri))
    else:
        server.core.tracklist.clear()
        logging.debug('got special uris: %r', [t['uri'] for t in hits])
        server.core.tracklist.add(uris=[t['uri'] for t in hits])
        server.core.playback.play()