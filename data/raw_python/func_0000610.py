def run_from_argv(self, argv):
        """
        Set up any environment changes requested (e.g., Python path
        and Django settings), then run this command. If the
        command raises a ``CommandError``, intercept it and print it sensibly
        to stderr.
        """
        parser = self.create_parser(argv[0], argv[1])
        args = parser.parse_args(argv[2:])
        handle_default_options(args)
        try:
            self.execute(args)
        except Exception as e:
            # self.stderr is not guaranteed to be set here
            try:
                fallback_stderr = OutputWrapper(sys.stderr, self.style.ERROR)
            except:
                fallback_stderr = self.stdout
            stderr = getattr(self, 'stderr', fallback_stderr)
            if args.traceback:
                stderr.write(traceback.format_exc())
            else:
                stderr.write('%s: %s' % (e.__class__.__name__, e))
            sys.exit(1)