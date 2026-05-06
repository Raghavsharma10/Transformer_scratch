def solve_recaptcha(self, google_key, page_url, timeout = 15 * 60):
		'''
		Solve a recaptcha on page `page_url` with the input value `google_key`.
		Timeout is `timeout` seconds, defaulting to 60 seconds.

		Return value is either the `g-recaptcha-response` value, or an exceptionj is raised
		(generally `CaptchaSolverFailure`)
		'''

		proxy = SocksProxy.ProxyLauncher([TWOCAPTCHA_IP])

		try:
			captcha_id = self.doGet('input', {
						'key'         : self.api_key,
						'method'      : "userrecaptcha",
						'googlekey'   : google_key,
						'pageurl'     : page_url,

						'proxy'       : proxy.get_wan_address(),
						'proxytype'   : "SOCKS5",

						'json'        : True,
					}
				)

			# Allow 15 minutes for the solution
			# I've been seeing times up to 160+ seconds in testing.
			return self._getresult(captcha_id=captcha_id, timeout=timeout)
		finally:
			proxy.stop()