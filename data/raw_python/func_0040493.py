def from_zip(cls, src='/tmp/app.zip', dest='/app'):
    """
    Unzips a zipped app project file and instantiates it.

    :param src: zipfile path
    :param dest: destination folder to extract the zipfile content

    Returns
      A project instance.
    """
    try:
      zf = zipfile.ZipFile(src, 'r')
    except FileNotFoundError:
      raise errors.InvalidPathError(src)
    except zipfile.BadZipFile:
      raise errors.InvalidZipFileError(src)
    [zf.extract(file, dest) for file in zf.namelist()]
    zf.close()
    return cls.from_path(dest)