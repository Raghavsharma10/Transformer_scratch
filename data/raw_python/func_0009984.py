def get_current_url(self):
		'''
		Probe the remote session for the current window URL.

		This is primarily used to do things like unwrap redirects,
		or circumvent outbound url wrappers.

		'''
		res = self.Page_getNavigationHistory()
		assert 'result' in res
		assert 'currentIndex' in res['result']
		assert 'entries' in res['result']

		return res['result']['entries'][res['result']['currentIndex']]['url']