def samReferencesToStr(filenameOrSamfile, indent=0):
    """
    List SAM/BAM file reference names and lengths.

    @param filenameOrSamfile: Either a C{str} SAM/BAM file name or an
        instance of C{pysam.AlignmentFile}.
    @param indent: An C{int} number of spaces to indent each line.
    @return: A C{str} describing known reference names and their lengths.
    """
    indent = ' ' * indent

    def _references(sam):
        result = []
        for i in range(sam.nreferences):
            result.append('%s%s (length %d)' % (
                indent, sam.get_reference_name(i), sam.lengths[i]))
        return '\n'.join(result)

    if isinstance(filenameOrSamfile, six.string_types):
        with samfile(filenameOrSamfile) as sam:
            return _references(sam)
    else:
        return _references(sam)