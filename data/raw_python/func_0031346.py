def main(args=None):
    """Script entry point."""

    if args is None:
        parser = get_argument_parser()
        args = parser.parse_args()

    #series_matrix_file = newstr(args.series_matrix_file, 'utf-8')
    #output_file = newstr(args.output_file, 'utf-8')
    #encoding = newstr(args.encoding, 'utf-8')
    series_matrix_file = args.series_matrix_file
    output_file = args.output_file
    encoding = args.encoding

    # log_file = args.log_file
    # quiet = args.quiet
    # verbose = args.verbose

    # logger = misc.get_logger(log_file = log_file, quiet = quiet,
    #        verbose = verbose)

    accessions, titles, celfile_urls = read_series_matrix(
        series_matrix_file, encoding=encoding)
    write_sample_sheet(output_file, accessions, titles, celfile_urls)

    return 0