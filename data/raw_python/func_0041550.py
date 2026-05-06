def configure_basic_logging(self, main_module_name, **kwargs):
        '''Use common logging options to configure all logging.

        Basic logging configuration is used to set levels for all logs from the main module and to
        filter out logs from other modules unless they are of one level in priority higher.

        :param main_module_name: name of the primary module for normal logging
        '''
        if not self._log_options_parent:
            raise ValueError('Missing log_options_parent')

        options = self[self._log_options_parent]
        log_level_index = LOG_LEVELS.index(options.log_level)
        log_kwargs = {
            'level': getattr(logging, options.log_level.upper()),
            'format': '[%(asctime)s #%(process)d] %(levelname)-8s %(name)-12s %(message)s',
            'datefmt': '%Y-%m-%dT%H:%M:%S%z',
        }

        if options.log_file == 'STDERR':
            log_kwargs['stream'] = sys.stderr
        elif options.log_file == 'STDOUT':
            log_kwargs['stream'] = sys.stdout
        else:
            log_kwargs['filename'] = options.log_file

        log_kwargs.update(kwargs)  # allow overrides from caller
        logging.basicConfig(**log_kwargs)

        # now filter out any other module's logging unless it's one level above the main
        other_log_level = getattr(logging, LOG_LEVELS[log_level_index + 1].upper())
        other_filter = OtherLoggingFilter(main_module_name, other_log_level)
        for handler in logging.root.handlers:
            handler.addFilter(other_filter)