def download_file(url, filename=None, show_progress=draw_pbar):
    '''
    Download a file and show progress

    url: the URL of the file to download
    filename: the filename to download it to (if not given, uses the url's filename part)
    show_progress: callback function to update a progress bar

    the show_progress function shall take two parameters: `seen` and `size`, and
    return nothing.

    This function returns the filename it has written the result to.
    '''
    if filename is None:
        filename = url.split('/')[-1]
    r = requests.get(url, stream=True)
    size = int(r.headers['Content-Length'].strip())
    seen = 0
    show_progress(0, size)
    seen = 1024
    with open(filename, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            seen += 1024
            show_progress(seen, size)
            if chunk:
                f.write(chunk)
                f.flush()
    return filename