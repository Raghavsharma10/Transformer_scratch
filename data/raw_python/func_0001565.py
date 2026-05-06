def parseArgv():
    """
    Command line option parser.
    """
    parser = OptionParser()
    parser.usage = r""" cat <TEXT> | %prog [--unit <UNIT>] [--output <SA_FILE>]

Create the suffix array of TEXT with the processing UNIT and optionally store it in SA_FILE for subsequent use.
UNIT may be set to 'byte', 'character' (given an encoding with the --encoding option) or 'word', which is the default.
"""

    parser.add_option("-i", "--input",
                      action="store", type="string", dest="input",
                      default=False,
                      help="Path of the file containing the input text. When '-' is given, read the standard input (default). If the path ends with '.gz', reads the decompressed file.")

    parser.add_option("-o", "--output",
                      action="store", type="string", dest="output",
                      default=False,
                      help="Store the suffix array of the input to the file OUTPUT. When '-' is given, writes to the standard output. If the filename ends with '.gz', the suffix array will be stored  compressed.")

    parser.add_option("", "--load",
                      action="store", type="string", dest="SAFile",
                      default=False,
                      help="Load a suffix array from SAFILE, this option and --input are mutually exclusive.")

    parser.add_option("-u", "--unit",
                      action="store", type="string", dest="unit",
                      default=DEFAULT_UNIT_STR,
                      help="Processing unit used for the creation of the suffix array." + \
                           'Possible values are "byte", "character" and "word". Default is "%s".' % DEFAULT_UNIT_STR + \
                           "This option is ignored when the suffix array is loaded from SAFILE." + \
                           'For characters, the input is decoded according to the encoding set via the option --encoding.')

    parser.add_option("-e", "--encoding",
                      action="store", type="string", dest="encoding",
                      default=DEFAULT_ENCODING,
                      help="Encoding of the input. This information is required only when processing characters. Default is '%s'." % DEFAULT_ENCODING)

    parser.add_option("-p", "--print",
                      action="store_true", dest="printSA",
                      default=False,
                      help="Prints the suffix array in a human readable format to the standard error output.")

    parser.add_option("", "--verbose",
                      action="store_true", dest="verbose",
                      default=False,
                      help="Prints more information.")

    parser.add_option("", "--no-lcps",
                      action="store_true", dest="noLCPs",
                      default=False,
                      help="Switch off the computation of LCPs. By doing so, the find functions are unusable.")

    (options, args) = parser.parse_args(_argv)
    strings = args[1:]
    return (options, strings)