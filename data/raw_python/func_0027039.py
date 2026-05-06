def dummy_image(filetype='gif'):
    """ Generate empty image in temporary file for testing """
    # 1x1px Transparent GIF
    GIF = 'R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7'
    tmp_file = tempfile.NamedTemporaryFile(suffix='.%s' % filetype)
    tmp_file.write(base64.b64decode(GIF))
    return open(tmp_file.name, 'rb')