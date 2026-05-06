def main():
    """Execute the converter using parameters provided on the command line"""

    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--outfile', metavar='output_file_path',
                        help="Save calltree stats to <outfile>")
    parser.add_argument('-i', '--infile', metavar='input_file_path',
                        help="Read Python stats from <infile>")
    parser.add_argument('-k', '--kcachegrind',
                        help="Run the kcachegrind tool on the converted data",
                        action="store_true")
    parser.add_argument('-r', '--run-script',
                        nargs=argparse.REMAINDER,
                        metavar=('scriptfile', 'args'),
                        dest='script',
                        help="Name of the Python script to run to collect"
                        " profiling data")
    args = parser.parse_args()

    outfile = args.outfile

    if args.script is not None:
        # collect profiling data by running the given script
        if not args.outfile:
            outfile = '%s.log' % os.path.basename(args.script[0])

        fd, tmp_path = tempfile.mkstemp(suffix='.prof', prefix='pyprof2calltree')
        os.close(fd)
        try:
            cmd = [
                sys.executable,
                '-m', 'cProfile',
                '-o', tmp_path,
            ]
            cmd.extend(args.script)
            subprocess.check_call(cmd)

            kg = CalltreeConverter(tmp_path)
        finally:
            os.remove(tmp_path)

    elif args.infile is not None:
        # use the profiling data from some input file
        if not args.outfile:
            outfile = '%s.log' % os.path.basename(args.infile)

        if args.infile == outfile:
            # prevent name collisions by appending another extension
            outfile += ".log"

        kg = CalltreeConverter(pstats.Stats(args.infile))

    else:
        # at least an input file or a script to run is required
        parser.print_usage()
        sys.exit(2)

    if args.outfile is not None or not args.kcachegrind:
        # user either explicitly required output file or requested by not
        # explicitly asking to launch kcachegrind
        sys.stderr.write("writing converted data to: %s\n" % outfile)
        with open(outfile, 'w') as f:
            kg.output(f)

    if args.kcachegrind:
        sys.stderr.write("launching kcachegrind\n")
        kg.visualize()