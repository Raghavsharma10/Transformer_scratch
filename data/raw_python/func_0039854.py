def run(self, argv=None):
        """
        Run the command-line application.

        This will dispatch to the specified function or raise a
        ``SystemExit`` and output the appropriate usage information
        if there is an error parsing the arguments.

        The default ``argv`` is equivalent to ``sys.argv[1:]``.

        """
        if argv is None:  # pragma: no cover
            argv = sys.argv[1:]
        argv = [str(v) for v in argv]
        meta, args = parse_args(self, argv)
        context_factory = contextmanager(make_generator(self.context_factory))
        with context_factory(self, args, **meta.context_kwargs) as context:
            main = load_main(meta)
            return main(context, args) or 0