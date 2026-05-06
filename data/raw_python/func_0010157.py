def write_file(data, outfilename):
    """Write a single file to disk."""
    if not data:
        return False
    try:
        with open(outfilename, 'w') as outfile:
            for line in data:
                if line:
                    outfile.write(line)
        return True
    except (OSError, IOError) as err:
        sys.stderr.write('An error occurred while writing {0}:\n{1}'
                         .format(outfilename, str(err)))
        return False