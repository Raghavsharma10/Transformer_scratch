def convert_md_to_rst(source, destination=None, backup_dir=None):
    """Try to convert the source, an .md (markdown) file, to an .rst
    (reStructuredText) file at the destination. If the destination isn't
    provided, it defaults to be the same as the source path except for the
    filename extension. If the destination file already exists, it will be
    overwritten. In the event of an error, the destination file will be
    left untouched."""

    # Doing this in the function instead of the module level ensures the
    # error occurs when the function is called, rather than when the module
    # is evaluated.
    try:
        import pypandoc
    except ImportError:
        # Don't give up right away; first try to install the python module.
        os.system("pip install pypandoc")
        import pypandoc

    # Set our destination path to a default, if necessary
    destination = destination or (os.path.splitext(source)[0] + '.rst')

    # Likewise for the backup directory
    backup_dir = backup_dir or os.path.join(os.path.dirname(destination),
                                            'bak')

    bak_name = (os.path.basename(destination) +
                time.strftime('.%Y%m%d%H%M%S.bak'))
    bak_path = os.path.join(backup_dir, bak_name)

    # If there's already a file at the destination path, move it out of the
    # way, but don't delete it.
    if os.path.isfile(destination):
        if not os.path.isdir(os.path.dirname(bak_path)):
            os.mkdir(os.path.dirname(bak_path))
        os.rename(destination, bak_path)

    try:
        # Try to convert the file.
        pypandoc.convert(
            source,
            'rst',
            format='md',
            outputfile=destination
        )
    except:
        # If for any reason the conversion fails, try to put things back
        # like we found them.
        if os.path.isfile(destination):
            os.remove(destination)
        if os.path.isfile(bak_path):
            os.rename(bak_path, destination)
        raise