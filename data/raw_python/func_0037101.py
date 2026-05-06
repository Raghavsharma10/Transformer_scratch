def _generate_filenames(sources):
    """Generate filenames.

    :param tuple sources: Sequence of strings representing path to file(s).
    :return: Path to file(s).
    :rtype: :py:class:`str`
    """

    for source in sources:

        if os.path.isdir(source):
            for path, dirlist, filelist in os.walk(source):
                for fname in filelist:
                    if nmrstarlib.VERBOSE:
                        print("Processing file: {}".format(os.path.abspath(fname)))
                    if GenericFilePath.is_compressed(fname):
                        if nmrstarlib.VERBOSE:
                            print("Skipping compressed file: {}".format(os.path.abspath(fname)))
                        continue
                    else:
                        yield os.path.join(path, fname)

        elif os.path.isfile(source):
            yield source

        elif GenericFilePath.is_url(source):
            yield source

        elif source.isdigit():
            try:
                urlopen(nmrstarlib.BMRB_REST + source)
                yield nmrstarlib.BMRB_REST + source
            except HTTPError:
                urlopen(nmrstarlib.PDB_REST + source + ".cif")
                yield nmrstarlib.PDB_REST + source + ".cif"

        elif re.match("[\w\d]{4}", source):
            yield nmrstarlib.PDB_REST + source + ".cif"

        else:
            raise TypeError("Unknown file source.")