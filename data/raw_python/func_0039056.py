def solve_simple_captcha(self, pathfile=None, filedata=None, filename=None):
		"""
		Upload a image (from disk or a bytearray), and then
		block until the captcha has been solved.
		Return value is the captcha result.

		either pathfile OR filedata AND filename should be specified.

		Failure will result in a subclass of WebRequest.CaptchaSolverFailure being
		thrown.
		"""

		captcha_id = self._submit(pathfile, filedata, filename)
		return self._getresult(captcha_id=captcha_id)