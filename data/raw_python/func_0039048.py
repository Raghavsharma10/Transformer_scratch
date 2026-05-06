def addSeleniumCookie(self, cookieDict):
		'''
		Install a cookie exported from a selenium webdriver into
		the active opener
		'''
		# print cookieDict
		cookie = http.cookiejar.Cookie(
				version            = 0,
				name               = cookieDict['name'],
				value              = cookieDict['value'],
				port               = None,
				port_specified     = False,
				domain             = cookieDict['domain'],
				domain_specified   = True,
				domain_initial_dot = False,
				path               = cookieDict['path'],
				path_specified     = False,
				secure             = cookieDict['secure'],
				expires            = cookieDict['expiry'] if 'expiry' in cookieDict else None,
				discard            = False,
				comment            = None,
				comment_url        = None,
				rest               = {"httponly":"%s" % cookieDict['httponly'] if 'httponly' in cookieDict else False},
				rfc2109            = False
			)

		self.addCookie(cookie)