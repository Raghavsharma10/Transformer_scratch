def downlad_file(url, fname):
    """Download file from url and save as fname."""
    print("Downloading {} as {}".format(url, fname))
    response = urlopen(url)
    download = response.read()
    with open(fname, 'wb') as fh:
        fh.write(download)