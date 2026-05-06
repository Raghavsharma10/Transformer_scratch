def main(argv=sys.argv, stream=sys.stderr):
    """Entry point for ``tappy`` command."""
    args = parse_args(argv)
    suite = build_suite(args)
    runner = unittest.TextTestRunner(verbosity=args.verbose, stream=stream)
    result = runner.run(suite)

    return get_status(result)