def _configure(self, config):
		"""Prepare the incoming configuration and ensure certain expected values are present.
		
		For example, this ensures BaseExtension is included in the extension list, and populates the logging config.
		"""
		config = config or dict()
		
		# We really need this to be there.
		if 'extensions' not in config: config['extensions'] = list()
		
		if not any(isinstance(ext, BaseExtension) for ext in config['extensions']):
			# Always make sure the BaseExtension is present since request/response objects are handy.
			config['extensions'].insert(0, BaseExtension())
		
		if not any(isinstance(ext, arguments.ArgumentExtension) for ext in config['extensions']):
			# Prepare a default set of argument mutators.
			config['extensions'].extend([
					arguments.ValidateArgumentsExtension(),
					arguments.ContextArgsExtension(),
					arguments.RemainderArgsExtension(),
					arguments.QueryStringArgsExtension(),
					arguments.FormEncodedKwargsExtension(),
					arguments.JSONKwargsExtension(),
				])
		
		config['extensions'].append(self)  # Allow the application object itself to register callbacks.
		
		try:
			addLoggingLevel('trace', logging.DEBUG - 5)
		except AttributeError:
			pass
		
		# Tests are skipped on these as we have no particular need to test Python's own logging mechanism.
		level = config.get('logging', {}).get('level', None)
		if level:  # pragma: no cover
			logging.basicConfig(level=getattr(logging, level.upper()))
		elif 'logging' in config:  # pragma: no cover
			logging.config.dictConfig(config['logging'])
		
		return config