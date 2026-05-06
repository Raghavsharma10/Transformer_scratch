def _listen_for_dweets_from_response(response):
    """Yields dweets as received from dweet.io's streaming API
    """
    streambuffer = ''
    for byte in response.iter_content():
        if byte:
            streambuffer += byte.decode('ascii')
            try:
                dweet = json.loads(streambuffer.splitlines()[1])
            except (IndexError, ValueError):
                continue
            if isstr(dweet):
                yield json.loads(dweet)
            streambuffer = ''