def unzip(archive, destination, filenames=None):
    """Unzip a zip archive into destination directory.

    It unzips either the whole archive or specific file(s) from the archive.

    Usage:
        >>> output = os.path.join(os.getcwd(), 'output')
        >>> # Archive can be an instance of a ZipFile class
        >>> archive = zipfile.ZipFile('test.zip', 'r')
        >>> # Or just a filename
        >>> archive = 'test.zip'
        >>> # Extracts all files
        >>> unzip(archive, output)
        >>> # Extract only one file
        >>> unzip(archive, output, 'my_file.txt')
        >>> # Extract a list of files
        >>> unzip(archive, output, ['my_file1.txt', 'my_file2.txt'])
        >>> unzip_file('test.zip', 'my_file.txt', output)

    Args:
        archive (zipfile.ZipFile or str): Zipfile object to extract from or
            path to the zip archive.
        destination (str): Path to the output directory.
        filenames (str or list of str or None): Path(s) to the filename(s)
            inside the zip archive that you want to extract.
    """
    close = False
    try:
        if not isinstance(archive, zipfile.ZipFile):
            archive = zipfile.ZipFile(archive, "r", allowZip64=True)
            close = True
        logger.info("Extracting: %s -> %s" % (archive.filename, destination))
        if isinstance(filenames, str):
            filenames = [filenames]
        if filenames is None:  # extract all
            filenames = archive.namelist()
        for filename in filenames:
            if filename.endswith("/"):  # it's a directory
                shell.mkdir(os.path.join(destination, filename))
            else:
                if not _extract_file(archive, destination, filename):
                    raise Exception()
        logger.info('Extracting zip archive "%s" succeeded' % archive.filename)
        return True
    except Exception:
        logger.exception("Error while unzipping archive %s" % archive.filename)
        return False
    finally:
        if close:
            archive.close()