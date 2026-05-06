def build_server_from_argparser(description=None, server_klass=None, handler_klass=None):
	"""
	Build a server from command line arguments. If a ServerClass or
	HandlerClass is specified, then the object must inherit from the
	corresponding AdvancedHTTPServer base class.

	:param str description: Description string to be passed to the argument parser.
	:param server_klass: Alternative server class to use.
	:type server_klass: :py:class:`.AdvancedHTTPServer`
	:param handler_klass: Alternative handler class to use.
	:type handler_klass: :py:class:`.RequestHandler`
	:return: A configured server instance.
	:rtype: :py:class:`.AdvancedHTTPServer`
	"""
	import argparse

	def _argp_dir_type(arg):
		if not os.path.isdir(arg):
			raise argparse.ArgumentTypeError("{0} is not a valid directory".format(repr(arg)))
		return arg

	def _argp_port_type(arg):
		if not arg.isdigit():
			raise argparse.ArgumentTypeError("{0} is not a valid port".format(repr(arg)))
		arg = int(arg)
		if arg < 0 or arg > 65535:
			raise argparse.ArgumentTypeError("{0} is not a valid port".format(repr(arg)))
		return arg

	description = (description or 'HTTP Server')
	server_klass = (server_klass or AdvancedHTTPServer)
	handler_klass = (handler_klass or RequestHandler)

	parser = argparse.ArgumentParser(conflict_handler='resolve', description=description, fromfile_prefix_chars='@')
	parser.epilog = 'When a config file is specified with --config only the --log, --log-file and --password options will be used.'
	parser.add_argument('-c', '--conf', dest='config', type=argparse.FileType('r'), help='read settings from a config file')
	parser.add_argument('-i', '--ip', dest='ip', default='0.0.0.0', help='the ip address to serve on')
	parser.add_argument('-L', '--log', dest='loglvl', choices=('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'), default='INFO', help='set the logging level')
	parser.add_argument('-p', '--port', dest='port', default=8080, type=_argp_port_type, help='port to serve on')
	parser.add_argument('-v', '--version', action='version', version=parser.prog + ' Version: ' + __version__)
	parser.add_argument('-w', '--web-root', dest='web_root', default='.', type=_argp_dir_type, help='path to the web root directory')
	parser.add_argument('--log-file', dest='log_file', help='log information to a file')
	parser.add_argument('--no-threads', dest='use_threads', action='store_false', default=True, help='disable threading')
	parser.add_argument('--password', dest='password', help='password to use for basic authentication')
	ssl_group = parser.add_argument_group('ssl options')
	ssl_group.add_argument('--ssl-cert', dest='ssl_cert', help='the ssl cert to use')
	ssl_group.add_argument('--ssl-key', dest='ssl_key', help='the ssl key to use')
	ssl_group.add_argument('--ssl-version', dest='ssl_version', choices=[p[9:] for p in dir(ssl) if p.startswith('PROTOCOL_')], help='the version of ssl to use')
	arguments = parser.parse_args()

	logging.getLogger('').setLevel(logging.DEBUG)
	console_log_handler = logging.StreamHandler()
	console_log_handler.setLevel(getattr(logging, arguments.loglvl))
	console_log_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s"))
	logging.getLogger('').addHandler(console_log_handler)

	if arguments.log_file:
		main_file_handler = logging.handlers.RotatingFileHandler(arguments.log_file, maxBytes=262144, backupCount=5)
		main_file_handler.setLevel(logging.DEBUG)
		main_file_handler.setFormatter(logging.Formatter("%(asctime)s %(name)-30s %(levelname)-10s %(message)s"))
		logging.getLogger('').setLevel(logging.DEBUG)
		logging.getLogger('').addHandler(main_file_handler)

	if arguments.config:
		config = ConfigParser()
		config.readfp(arguments.config)
		server = build_server_from_config(
			config,
			'server',
			server_klass=server_klass,
			handler_klass=handler_klass
		)
	else:
		server = server_klass(
			handler_klass,
			address=(arguments.ip, arguments.port),
			use_threads=arguments.use_threads,
			ssl_certfile=arguments.ssl_cert,
			ssl_keyfile=arguments.ssl_key,
			ssl_version=arguments.ssl_version
		)
		server.serve_files_root = arguments.web_root

	if arguments.password:
		server.auth_add_creds('', arguments.password)
	return server