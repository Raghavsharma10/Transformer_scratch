def makeOuputDir(outputDir, force):
    """
    Create or check for an output directory.

    @param outputDir: A C{str} output directory name, or C{None}.
    @param force: If C{True}, allow overwriting of pre-existing files.
    @return: The C{str} output directory name.
    """
    if outputDir:
        if exists(outputDir):
            if not force:
                print('Will not overwrite pre-existing files. Use --force to '
                      'make me.', file=sys.stderr)
                sys.exit(1)
        else:
            mkdir(outputDir)
    else:
        outputDir = mkdtemp()
        print('Writing output files to %s' % outputDir)

    return outputDir