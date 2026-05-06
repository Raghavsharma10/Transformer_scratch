def start(track_file,
          twitter_api_key,
          twitter_api_secret,
          twitter_access_token,
          twitter_access_token_secret,
          poll_interval=15,
          unfiltered=False,
          languages=None,
          debug=False,
          outfile=None):
    """Start the stream."""
    listener = construct_listener(outfile)
    checker = BasicFileTermChecker(track_file, listener)

    auth = get_tweepy_auth(twitter_api_key,
                           twitter_api_secret,
                           twitter_access_token,
                           twitter_access_token_secret)

    stream = DynamicTwitterStream(auth, listener, checker, unfiltered=unfiltered, languages=languages)

    set_terminate_listeners(stream)
    if debug:
        set_debug_listener(stream)

    begin_stream_loop(stream, poll_interval)