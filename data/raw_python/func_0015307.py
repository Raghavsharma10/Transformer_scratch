def run(cls):
        """Runs the whole cli:

        1. Registers console logging handler
        2. Creates argparser from all assistants and actions
        3. Parses args and decides what to run
        4. Runs a proper assistant or action
        """
        sigint_handler.override()
        # set settings.USE_CACHE before constructing parser, since constructing
        # parser requires loaded assistants
        settings.USE_CACHE = False if '--no-cache' in sys.argv else True
        cls.register_console_logging_handler(logger.logger)
        is_log_file = logger.add_log_file_handler(settings.LOG_FILE)
        if not is_log_file:
            logger.logger.warning("Could not create log file '{0}'.".format(settings.LOG_FILE))
        cls.inform_of_short_bin_name(sys.argv[0])
        top_assistant = bin.TopAssistant()
        tree = top_assistant.get_subassistant_tree()
        argparser = argparse_generator.ArgparseGenerator.\
            generate_argument_parser(tree, actions=actions.actions)
        parsed_args = vars(argparser.parse_args())

        parsed_args_decoded = dict()
        for k, v in parsed_args.items():
            parsed_args_decoded[k] = \
                v.decode(utils.defenc) if not six.PY3 and isinstance(v, str) else v
        parsed_args_decoded['__ui__'] = 'cli'

        if parsed_args.get('da_debug'):
            cls.change_logging_level(logging.DEBUG)

        # Prepare Action/PathRunner
        if actions.is_action_run(**parsed_args_decoded):
            to_run = actions.get_action_to_run(**parsed_args_decoded)(**parsed_args_decoded)
        else:
            parsed_args = cls.transform_executable_assistant_alias(parsed_args_decoded)
            path = top_assistant.get_selected_subassistant_path(**parsed_args_decoded)
            to_run = path_runner.PathRunner(path, parsed_args_decoded)

        try:
            to_run.run()
        except exceptions.ExecutionException:
            # error is already logged, just catch it and silently exit here
            sys.exit(1)