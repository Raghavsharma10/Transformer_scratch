def unzip_file(filename, location, flatten=True):
    """Unzip the file (zip file located at filename) to the destination
    location"""
    if not os.path.exists(location):
        os.makedirs(location)
    zipfp = open(filename, 'rb')
    try:
        zip = zipfile.ZipFile(zipfp)
        leading = has_leading_dir(zip.namelist()) and flatten
        for name in zip.namelist():
            data = zip.read(name)
            fn = name
            if leading:
                fn = split_leading_dir(name)[1]
            fn = os.path.join(location, fn)
            dir = os.path.dirname(fn)
            if not os.path.exists(dir):
                os.makedirs(dir)
            if fn.endswith('/') or fn.endswith('\\'):
                # A directory
                if not os.path.exists(fn):
                    os.makedirs(fn)
            else:
                fp = open(fn, 'wb')
                try:
                    fp.write(data)
                finally:
                    fp.close()
    finally:
        zipfp.close()