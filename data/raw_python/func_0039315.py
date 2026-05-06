def solve_simple_captcha(self, pathfile=None, filedata=None, filename=None):
		"""
		Upload a image (from disk or a bytearray), and then
		block until the captcha has been solved.
		Return value is the captcha result.

		either pathfile OR filedata should be specified. Filename is ignored (and is
		only kept for compatibility with the 2captcha solver interface)

		Failure will result in a subclass of WebRequest.CaptchaSolverFailure being
		thrown.
		"""

		if pathfile and os.path.exists(pathfile):
			fp = open(pathfile, 'rb')
		elif filedata:
			fp = io.BytesIO(filedata)
		else:
			raise ValueError("You must pass either a valid file path, or a bytes array containing the captcha image!")

		try:
			task = python_anticaptcha.ImageToTextTask(fp)
			job = self.client.createTask(task)

			job.join(maximum_time = self.waittime)

			return job.get_captcha_text()

		except python_anticaptcha.AnticaptchaException as e:
			raise exc.CaptchaSolverFailure("Failure solving captcha: %s, %s, %s" % (
					e.error_id,
					e.error_code,
					e.error_description,
				))