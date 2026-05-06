def _init_config(self):
        """return command's configuration from call's arguments
        """
        options = self.parser.parse_args()
        if options.config is None and options.input is None:
            self.parser.print_help()
            sys.exit(2)

        if options.config is not None:
            configFactory = ConfigFactory()
            config = configFactory.load_from_file(options.config)
        else:
            config = ConfigObject()

        if options.input is not None:
            config["input"]["locations"] = [str(x) for x in options.input]
        if options.arguments is not None:
            config["input"]["arguments"] = dict((x.partition("=")[0], x.partition("=")[2]) for x in options.arguments)

        if options.output is not None:
            config["output"]["location"] = options.output

        if options.no_validate is not None:
            config["input"]["validate"] = not options.no_validate

        if options.dry_run is not None:
            self.dry_run = options.dry_run
        if options.watch is not None:
            self.watch = options.watch
        if options.traceback is not None:
            self.traceback = options.traceback

        if options.quiet is not None:
            self.logger.setLevel(logging.WARNING)
        if options.silence is not None:
            logging.disable(logging.CRITICAL)

        configService = ConfigService()
        configService.validate(config)
        self.config = config