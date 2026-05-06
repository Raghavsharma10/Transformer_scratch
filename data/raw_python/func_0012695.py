def download(url, target, headers=None, trackers=()):
    """Download a file using requests.

    This is like urllib.request.urlretrieve, but:

    - requests validates SSL certificates by default
    - you can pass tracker objects to e.g. display a progress bar or calculate
      a file hash.
    """
    if headers is None:
        headers = {}
    headers.setdefault('user-agent', 'requests_download/'+__version__)
    r = requests.get(url, headers=headers, stream=True)
    r.raise_for_status()
    for t in trackers:
        t.on_start(r)


    with open(target, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                for t in trackers:
                    t.on_chunk(chunk)

    for t in trackers:
        t.on_finish()