def find_all_files(self):
		"""
		Find files including those in subrepositories.
		"""
		files = self.find_files()
		subrepo_files = (
			posixpath.join(subrepo.location, filename)
			for subrepo in self.subrepos()
			for filename in subrepo.find_files()
		)
		return itertools.chain(files, subrepo_files)