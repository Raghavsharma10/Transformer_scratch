def execute_javascript(self, *args, **kwargs):
		'''
		Execute a javascript string in the context of the browser tab.
		'''

		ret = self.__exec_js(*args, **kwargs)
		return ret