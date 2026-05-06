def handler(self):
		'Handler function'
		from feedjack import filters # shouldn't be imported globally, as they may depend on models
		proc_func = getattr(filters, self.handler_name or self.name, None)
		if proc_func is None:
			if '.' not in self.handler_name:
				raise ImportError('Processing function not available: {0}'.format(self.handler_name))
			proc_module, proc_func = it.imap(str, self.handler_name.rsplit('.', 1))
			proc_func = getattr(__import__(proc_module, fromlist=[proc_func]), proc_func)
		return proc_func