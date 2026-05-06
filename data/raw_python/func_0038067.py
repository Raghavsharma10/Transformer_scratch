def _invoke(self, *params):
		"""
		Invoke self.exe as a subprocess
		"""
		cmd = [self.exe] + list(params)
		proc = subprocess.Popen(
			cmd, stdout=subprocess.PIPE,
			stderr=subprocess.PIPE, cwd=self.location, env=self.env)
		stdout, stderr = proc.communicate()
		if not proc.returncode == 0:
			raise RuntimeError(stderr.strip() or stdout.strip())
		return stdout.decode('utf-8')