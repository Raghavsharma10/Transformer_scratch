def pyle(argv=None):
    """Execute pyle with the specified arguments, or sys.argv if no arguments specified."""

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("-m", "--modules", dest="modules", action='append',
        help="import MODULE before evaluation. May be specified more than once.")
    parser.add_argument("-i", "--inplace", dest="inplace", action='store_true', default=False,
        help="edit files in place. When used with file name arguments, the files will be replaced by the output of the evaluation")
    parser.add_argument("-e", "--expression", action="append",
        dest="expressions", help="an expression to evaluate for each line")
    parser.add_argument('files', nargs='*',
        help="files to read as input. If used with --inplace, the files will be replaced with the output")
    parser.add_argument("--traceback", action="store_true", default=False,
        help="print a traceback on stderr when an expression fails for a line")

    args = parser.parse_args() if not argv else parser.parse_args(argv)

    pyle_evaluate(args.expressions, args.modules, args.inplace, args.files,
                  args.traceback)