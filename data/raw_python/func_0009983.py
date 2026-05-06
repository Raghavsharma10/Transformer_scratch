def set_cookie(self, cookie):
		'''
		Add a cookie to the remote chromium instance.

		Passed value `cookie` must be an instance of `http.cookiejar.Cookie()`.
		'''

		# Function path: Network.setCookie
		# Domain: Network
		# Method name: setCookie
		# WARNING: This function is marked 'Experimental'!
		# Parameters:
		#         Required arguments:
		#                 'url' (type: string) -> The request-URI to associate with the setting of the cookie. This value can affect the default domain and path values of the created cookie.
		#                 'name' (type: string) -> The name of the cookie.
		#                 'value' (type: string) -> The value of the cookie.
		#         Optional arguments:
		#                 'domain' (type: string) -> If omitted, the cookie becomes a host-only cookie.
		#                 'path' (type: string) -> Defaults to the path portion of the url parameter.
		#                 'secure' (type: boolean) -> Defaults ot false.
		#                 'httpOnly' (type: boolean) -> Defaults to false.
		#                 'sameSite' (type: CookieSameSite) -> Defaults to browser default behavior.
		#                 'expirationDate' (type: Timestamp) -> If omitted, the cookie becomes a session cookie.
		# Returns:
		#         'success' (type: boolean) -> True if successfully set cookie.

		# Description: Sets a cookie with the given cookie data; may overwrite equivalent cookies if they exist.

		assert isinstance(cookie, http.cookiejar.Cookie), 'The value passed to `set_cookie` must be an instance of http.cookiejar.Cookie().' + \
			' Passed: %s ("%s").' % (type(cookie), cookie)

		# Yeah, the cookielib stores this attribute as a string, despite it containing a
		# boolean value. No idea why.
		is_http_only = str(cookie.get_nonstandard_attr('httponly', 'False')).lower() == "true"


		# I'm unclear what the "url" field is actually for. A cookie only needs the domain and
		# path component to be fully defined. Considering the API apparently allows the domain and
		# path parameters to be unset, I think it forms a partially redundant, with some
		# strange interactions with mode-changing between host-only and more general
		# cookies depending on what's set where.
		# Anyways, given we need a URL for the API to work properly, we produce a fake
		# host url by building it out of the relevant cookie properties.
		fake_url = urllib.parse.urlunsplit((
				"http" if is_http_only else "https",  # Scheme
				cookie.domain,                        # netloc
				cookie.path,                          # path
				'',                                   # query
				'',                                   # fragment
			))

		params = {
				'url'      : fake_url,

				'name'     : cookie.name,
				'value'    : cookie.value if cookie.value else "",
				'domain'   : cookie.domain,
				'path'     : cookie.path,
				'secure'   : cookie.secure,
				'expires'  : float(cookie.expires) if cookie.expires else float(2**32),

				'httpOnly' : is_http_only,

				# The "sameSite" flag appears to be a chromium-only extension for controlling
				# cookie sending in non-first-party contexts. See:
				# https://bugs.chromium.org/p/chromium/issues/detail?id=459154
				# Anyways, we just use the default here, whatever that is.
				# sameSite       = cookie.xxx
			}

		ret = self.Network_setCookie(**params)

		return ret