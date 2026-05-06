def install_from_zip(url):
    """Download and unzip from url."""
    fname = 'tmp.zip'
    downlad_file(url, fname)
    unzip_file(fname)
    print("Removing {}".format(fname))
    os.unlink(fname)