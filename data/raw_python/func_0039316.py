def solve_recaptcha(self, google_key, page_url, timeout = 15 * 60):
		'''
		Solve a recaptcha on page `page_url` with the input value `google_key`.
		Timeout is `timeout` seconds, defaulting to 60 seconds.

		Return value is either the `g-recaptcha-response` value, or an exceptionj is raised
		(generally `CaptchaSolverFailure`)
		'''

		proxy = SocksProxy.ProxyLauncher(ANTICAPTCHA_IPS)

		try:
			antiprox = python_anticaptcha.Proxy(
					proxy_type     = "socks5",
					proxy_address  = proxy.get_wan_ip(),
					proxy_port     = proxy.get_wan_port(),
					proxy_login    = None,
					proxy_password = None,
				)

			task = python_anticaptcha.NoCaptchaTask(
					website_url = page_url,
					website_key = google_key,
					proxy       = antiprox,
					user_agent  = dict(self.wg.browserHeaders).get('User-Agent')
				)
			job = self.client.createTask(task)
			job.join(maximum_time = timeout)

			return job.get_solution_response()
		except python_anticaptcha.AnticaptchaException as e:
			raise exc.CaptchaSolverFailure("Failure solving captcha: %s, %s, %s" % (
					e.error_id,
					e.error_code,
					e.error_description,
				))

		finally:
			proxy.stop()