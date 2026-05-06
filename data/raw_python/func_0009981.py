def update_headers(self, header_args):
		'''
		Given a set of headers, update both the user-agent
		and additional headers for the remote browser.

		header_args must be a dict. Keys are the names of
		the corresponding HTTP header.

		return value is a 2-tuple of the results of the user-agent
		update, as well as the extra headers update.
		If no 'User-Agent' key is present in the new headers,
		the first item in the tuple will be None

		'''
		assert isinstance(header_args, dict), "header_args must be a dict, passed type was %s" \
			% (type(header_args), )

		ua = header_args.pop('User-Agent', None)
		ret_1 = None
		if ua:
			ret_1 = self.Network_setUserAgentOverride(userAgent=ua)


		ret_2 = self.Network_setExtraHTTPHeaders(headers = header_args)

		return (ret_1, ret_2)