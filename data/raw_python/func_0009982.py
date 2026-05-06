def get_cookies(self):
		'''
		Retreive the cookies from the remote browser.

		Return value is a list of http.cookiejar.Cookie() instances.
		These can be directly used with the various http.cookiejar.XXXCookieJar
		cookie management classes.
		'''
		ret = self.Network_getAllCookies()

		assert 'result' in ret, "No return value in function response!"
		assert 'cookies' in ret['result'], "No 'cookies' key in function response"

		cookies = []
		for raw_cookie in ret['result']['cookies']:

			# Chromium seems to support the following key values for the cookie dict:
			# 	"name"
			# 	"value"
			# 	"domain"
			# 	"path"
			# 	"expires"
			# 	"httpOnly"
			# 	"session"
			# 	"secure"
			#
			#  This seems supported by the fact that the underlying chromium cookie implementation has
			#  the following members:
			#        std::string name_;
			#        std::string value_;
			#        std::string domain_;
			#        std::string path_;
			#        base::Time creation_date_;
			#        base::Time expiry_date_;
			#        base::Time last_access_date_;
			#        bool secure_;
			#        bool httponly_;
			#        CookieSameSite same_site_;
			#        CookiePriority priority_;
			#
			# See chromium/net/cookies/canonical_cookie.h for more.
			#
			# I suspect the python cookie implementation is derived exactly from the standard, while the
			# chromium implementation is more of a practically derived structure.

			# Network.setCookie

			baked_cookie = http.cookiejar.Cookie(
					# We assume V0 cookies, principally because I don't think I've /ever/ actually encountered a V1 cookie.
					# Chromium doesn't seem to specify it.
					version            = 0,

					name               = raw_cookie['name'],
					value              = raw_cookie['value'],
					port               = None,
					port_specified     = False,
					domain             = raw_cookie['domain'],
					domain_specified   = True,
					domain_initial_dot = False,
					path               = raw_cookie['path'],
					path_specified     = False,
					secure             = raw_cookie['secure'],
					expires            = raw_cookie['expires'],
					discard            = raw_cookie['session'],
					comment            = None,
					comment_url        = None,
					rest               = {"httponly":"%s" % raw_cookie['httpOnly']},
					rfc2109            = False
				)
			cookies.append(baked_cookie)

		return cookies