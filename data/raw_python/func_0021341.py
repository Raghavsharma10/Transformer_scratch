def build_server_from_config(config, section_name, server_klass=None, handler_klass=None):
	"""
	Build a server from a provided :py:class:`configparser.ConfigParser`
	instance. If a ServerClass or HandlerClass is specified, then the
	object must inherit from the corresponding AdvancedHTTPServer base
	class.

	:param config: Configuration to retrieve settings from.
	:type config: :py:class:`configparser.ConfigParser`
	:param str section_name: The section name of the configuration to use.
	:param server_klass: Alternative server class to use.
	:type server_klass: :py:class:`.AdvancedHTTPServer`
	:param handler_klass: Alternative handler class to use.
	:type handler_klass: :py:class:`.RequestHandler`
	:return: A configured server instance.
	:rtype: :py:class:`.AdvancedHTTPServer`
	"""
	server_klass = (server_klass or AdvancedHTTPServer)
	handler_klass = (handler_klass or RequestHandler)
	port = config.getint(section_name, 'port')
	web_root = None
	if config.has_option(section_name, 'web_root'):
		web_root = config.get(section_name, 'web_root')

	if config.has_option(section_name, 'ip'):
		ip = config.get(section_name, 'ip')
	else:
		ip = '0.0.0.0'
	ssl_certfile = None
	if config.has_option(section_name, 'ssl_cert'):
		ssl_certfile = config.get(section_name, 'ssl_cert')
	ssl_keyfile = None
	if config.has_option(section_name, 'ssl_key'):
		ssl_keyfile = config.get(section_name, 'ssl_key')
	ssl_version = None
	if config.has_option(section_name, 'ssl_version'):
		ssl_version = config.get(section_name, 'ssl_version')
	server = server_klass(
		handler_klass,
		address=(ip, port),
		ssl_certfile=ssl_certfile,
		ssl_keyfile=ssl_keyfile,
		ssl_version=ssl_version
	)

	if config.has_option(section_name, 'password_type'):
		password_type = config.get(section_name, 'password_type')
	else:
		password_type = 'md5'
	if config.has_option(section_name, 'password'):
		password = config.get(section_name, 'password')
		if config.has_option(section_name, 'username'):
			username = config.get(section_name, 'username')
		else:
			username = ''
		server.auth_add_creds(username, password, pwtype=password_type)
	cred_idx = 0
	while config.has_option(section_name, 'password' + str(cred_idx)):
		password = config.get(section_name, 'password' + str(cred_idx))
		if not config.has_option(section_name, 'username' + str(cred_idx)):
			break
		username = config.get(section_name, 'username' + str(cred_idx))
		server.auth_add_creds(username, password, pwtype=password_type)
		cred_idx += 1

	if web_root is None:
		server.serve_files = False
	else:
		server.serve_files = True
		server.serve_files_root = web_root
		if config.has_option(section_name, 'list_directories'):
			server.serve_files_list_directories = config.getboolean(section_name, 'list_directories')
	return server