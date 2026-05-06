def load(self, filename, replace = False):
		''' Loads a configuration file (JSON). '''

		import os, json, re
		if os.path.exists(filename):
			f = open(filename, 'r')
			content = f.read()
			content = re.sub('[\t ]*?[#].*?\n', '', content)
			try:
				settings = json.loads(content)
			except ValueError:
				# This means that the configuration file is not a valid JSON document
				from lltk.exceptions import ConfigurationError
				raise ConfigurationError('\'' + filename + '\' is not a valid JSON document.')
			f.close()
			if replace:
				self.settings = settings
			else:
				self.settings.update(settings)
		else:
			lltkfilename = self.settings['module-path'] + '/' + self.settings['lltk-config-path'] + filename
			if os.path.exists(lltkfilename):
				# This means that filename was provided relative to the lltk module path
				return self.load(lltkfilename)
			from lltk.exceptions import ConfigurationError
			raise ConfigurationError('\'' + filename + '\' seems to be non-existent.')