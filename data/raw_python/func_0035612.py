def _main():
    """ Command line interface for testing.
    """
    import pprint
    import tempfile

    try:
        image = sys.argv[1]
    except IndexError:
        print("Usage: python -m pyrobase.webservice.imgur <url>")
    else:
        try:
            pprint.pprint(copy_image_from_url(image, cache_dir=tempfile.gettempdir()))
        except UploadError as exc:
            print("Upload error. %s" % exc)