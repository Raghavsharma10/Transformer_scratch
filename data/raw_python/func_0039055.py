def _submit(self, pathfile, filedata, filename):
		'''
		Submit either a file from disk, or a in-memory file to the solver service, and
		return the request ID associated with the new captcha task.
		'''
		if pathfile and os.path.exists(pathfile):
			files = {'file': open(pathfile, 'rb')}
		elif filedata:
			assert filename
			files = {'file' : (filename, io.BytesIO(filedata))}
		else:
			raise ValueError("You must pass either a valid file path, or a bytes array containing the captcha image!")

		payload = {
			'key'    : self.api_key,
			'method' : 'post',
			'json'   : True,
			}

		self.log.info("Uploading to 2Captcha.com.")

		url = self.getUrlFor('input', {})

		request = requests.post(url, files=files, data=payload)

		if not request.ok:
			raise exc.CaptchaSolverFailure("Posting captcha to solve failed!")

		resp_json = json.loads(request.text)
		return self._process_response(resp_json)