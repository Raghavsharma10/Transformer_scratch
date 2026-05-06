def _setup_component(self, storm_conf, context):
        """Add helpful instance variables to component after initial handshake
        with Storm.  Also configure logging.
        """
        self.topology_name = storm_conf.get("topology.name", "")
        self.task_id = context.get("taskid", "")
        self.component_name = context.get("componentid")
        # If using Storm before 0.10.0 componentid is not available
        if self.component_name is None:
            self.component_name = context.get("task->component", {}).get(
                str(self.task_id), ""
            )
        self.debug = storm_conf.get("topology.debug", False)
        self.storm_conf = storm_conf
        self.context = context

        # Set up logging
        self.logger = logging.getLogger(".".join((__name__, self.component_name)))
        log_path = self.storm_conf.get("pystorm.log.path")
        log_file_name = self.storm_conf.get(
            "pystorm.log.file",
            "pystorm_{topology_name}" "_{component_name}" "_{task_id}" "_{pid}.log",
        )
        root_log = logging.getLogger()
        log_level = self.storm_conf.get("pystorm.log.level", "info")
        if log_path:
            max_bytes = self.storm_conf.get("pystorm.log.max_bytes", 1000000)  # 1 MB
            backup_count = self.storm_conf.get("pystorm.log.backup_count", 10)
            log_file = join(
                log_path,
                (
                    log_file_name.format(
                        topology_name=self.topology_name,
                        component_name=self.component_name,
                        task_id=self.task_id,
                        pid=self.pid,
                    )
                ),
            )
            handler = RotatingFileHandler(
                log_file, maxBytes=max_bytes, backupCount=backup_count
            )
            log_format = self.storm_conf.get(
                "pystorm.log.format",
                "%(asctime)s - %(name)s - " "%(levelname)s - %(message)s",
            )
        else:
            self.log(
                "pystorm StormHandler logging enabled, so all messages at "
                'levels greater than "pystorm.log.level" ({}) will be sent'
                " to Storm.".format(log_level)
            )
            handler = StormHandler(self.serializer)
            log_format = self.storm_conf.get(
                "pystorm.log.format", "%(asctime)s - %(name)s - " "%(message)s"
            )
        formatter = logging.Formatter(log_format)
        log_level = _PYTHON_LOG_LEVELS.get(log_level, logging.INFO)
        if self.debug:
            # potentially override logging that was provided if
            # topology.debug was set to true
            log_level = logging.DEBUG
        handler.setLevel(log_level)
        handler.setFormatter(formatter)
        root_log.addHandler(handler)
        self.logger.setLevel(log_level)
        logging.getLogger("pystorm").setLevel(log_level)
        # Redirect stdout to ensure that print statements/functions
        # won't disrupt the multilang protocol
        if self.serializer.output_stream == sys.stdout:
            sys.stdout = LogStream(logging.getLogger("pystorm.stdout"))